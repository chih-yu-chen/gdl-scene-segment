import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
pkg_path = Path(__file__).parents[1]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net
from datasets.scannet_dataset import ScanNetDataset
from model import utils
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# parse arguments outside python
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="directory to the ScanNet dataset")
parser.add_argument("--gpu", type=str, default="0",
                    help="which gpu")
parser.add_argument("--cpu", action="store_true",
                    help="use cpu instead of gpu")
parser.add_argument("--evaluate", action="store_true",
                    help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, default = 'xyz',
                    help="'xyz', 'xyzrgb', or 'hks', default: xyz")
parser.add_argument("--preprocess", type=str,
                    help="which preprocessing", required=True)
parser.add_argument("--without_gradient_rotations", action="store_true",
                    help="without learned gradient rotations")
parser.add_argument("--experiment", type=str, required=True,
                    help="experiment name")
args = parser.parse_args()



# computing devices
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.gpu}')



# task settings
n_class = 20
class_names = "mIoU,\
wall,floor,cabinet,bed,chair,\
sofa,table,door,window,bookshelf,\
picture,counter,desk,curtain,refridgerator,\
shower curtain,toilet,sink,bathtub,otherfurniture\n"

# model settings
input_features = args.input_features # one of ['xyz', 'xyzrgb, 'hks']
k_eig = 128



# training settings
train = not args.evaluate
n_epoch = 200
pseudo_batch_size = 8
lr = 1e-3
lr_step_size = 50
checkpt_every = 10
augment_random_rotate = (input_features == 'xyz') | (input_features == 'xyzrgb')
with_rgb = (input_features == 'xyzrgb')
with_gradient_rotations = not args.without_gradient_rotations


# paths
experiment = args.experiment
data_dir = Path(args.data_dir)
op_cache_dir = data_dir/ "diffusion-net"/ f"op_cache_{k_eig}"
op_cache_dir.mkdir(parents=True, exist_ok=True)
exp_dir = Path("..", "experiments", experiment).resolve()
exp_dir.mkdir(parents=True, exist_ok=True)
model_path = exp_dir/ "model.pt"
pred_dir = exp_dir/ "preds"
pred_dir.mkdir(parents=True, exist_ok=True)



# datasets
test_dataset = ScanNetDataset(train=False, data_dir=data_dir, with_rgb=with_rgb, preprocess=args.preprocess, k_eig=k_eig, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

if train:
    train_dataset = ScanNetDataset(train=True, data_dir=data_dir, with_rgb=with_rgb, preprocess=args.preprocess, k_eig=k_eig, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# the model
C_in = {'xyz':3, 'xyzrgb': 6, 'hks':16}[input_features]

model = diffusion_net.layers.DiffusionNet(C_in=C_in, C_out=n_class,
                                          C_width=128, N_block=4,
                                          last_activation=lambda x: torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='vertices',
                                          dropout=True,
                                          with_gradient_rotations=with_gradient_rotations)

model = model.to(device)



# load the pretrained model
if not train:
    print(f"Loading pretrained model from: {model_path}")
    model.load_state_dict(torch.load(str(model_path)))



# the optimizer & learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# DiffusionNet human segmentation
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5, verbose=True)
# PicassoNet++ 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, verbose=True)
# VMNet & DGNet
# lr_lambda = lambda epoch: (1 - epoch/(n_epoch+1)) ** 0.9
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)



# the training epoch
def train_epoch():

    # set model to 'train' mode
    model.train()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    optimizer.zero_grad()

    for i, (verts, rgb, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, _, ref_idx) in enumerate(tqdm(train_loader)):

        # augmentation
        if augment_random_rotate:
            verts = utils.random_rotate_points_z(verts)
        verts = utils.random_translate(verts, scale=1)
        verts = utils.random_flip(verts)
        verts = utils.random_scale(verts, max_scale=50)

        # move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # rgb features
        if with_rgb:
            rgb = utils.random_rgb_jitter(rgb, scale=0.05)
            rgb = rgb.to(device)

        # construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'xyzrgb':
            features = torch.hstack((verts, rgb))
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        total_loss += loss.item()
        loss.backward()
        
        # track accuracy
        pred_labels = torch.argmax(preds.cpu(), dim=1)
        pred_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, pred_labels)
        labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, labels.cpu())
        this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class)
        tps += this_tps
        fps += this_fps
        fns += this_fns

        # step the optimizer
        if (i+1) % pseudo_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    ious = tps / (tps+fps+fns)

    return total_loss/len(train_loader), np.insert(ious, 0, ious[1:].mean())



# the validation or test
def test(save=False):
    
    model.eval()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    with torch.no_grad():
    
        for verts, rgb, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene, ref_idx in tqdm(test_loader):

            # move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # rgb features
            if with_rgb:
                rgb = rgb.to(device)

            # construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'xyzrgb':
                features = torch.hstack((verts, rgb))
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track loss
            loss = torch.nn.functional.nll_loss(preds, labels)
            total_loss += loss.item()

            # track accuracy
            pred_labels = torch.argmax(preds.cpu(), dim=1)
            pred_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, pred_labels)
            labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, labels.cpu())
            this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class)
            tps += this_tps
            fps += this_fps
            fns += this_fns

            # save prediction
            if save:
                pred_labels[pred_labels == -100] = -1
                test_dataset.classes = np.append(test_dataset, [0])
                pred_labels = test_dataset.classes[pred_labels]
                np.savetxt(pred_dir/ f"{scene}_labels.txt", pred_labels, fmt='%d', delimiter='\n')
            
    ious = tps / (tps+fps+fns)

    return total_loss/len(test_loader), np.insert(ious, 0, ious[1:].mean())



# actual running
torch.set_printoptions(sci_mode=False)

if train:

    print("Training...")

    with open(model_path.with_name("train_iou.csv"), 'w') as f:
        f.write(class_names)
    with open(model_path.with_name("test_iou.csv"), 'w') as f:
        f.write(class_names)

    for epoch in range(n_epoch):

        train_loss, train_ious = train_epoch()
        test_loss, test_ious = test()
        scheduler.step()

        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_ious[0]}\n")
        print(f"Test Loss: {test_loss:.4f}, Test mIoU: {test_ious[0]}")

        with open(model_path.with_name("train_iou.csv"), 'ab') as f:
            np.savetxt(f, train_ious[np.newaxis,:], delimiter=',')
        with open(model_path.with_name("test_iou.csv"), 'ab') as f:
            np.savetxt(f, test_ious[np.newaxis,:], delimiter=',')
        with open(model_path.with_name("loss.csv"), 'a') as f:
            f.write(str(train_loss)+",")
            f.write(str(test_loss)+"\n")
        
        if (epoch+1) % checkpt_every == 0:
            torch.save(model.state_dict(), model_path.with_stem(f"checkpoint{epoch+1}"))
            print(" ==> model checkpoint saved")

    torch.save(model.state_dict(), model_path)
    print(" ==> last model saved")

test_loss, test_ious = test(save=True)
print(f"Overall Test Loss: {test_loss:.4f}, Test mIoU: {test_ious[0]}")
