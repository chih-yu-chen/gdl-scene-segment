import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
pkg_path = Path(__file__).parent/"diffusion-net"/"src"
sys.path.append(str(pkg_path))
import diffusion_net
from scannet_dataset import ScanNetDataset
import utils


# parse arguments outside python
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, help="which gpu")
parser.add_argument("--cpu", action="store_true", help="use cpu instead of gpu")
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="'xyz', 'xyzrgb', or 'hks', default: xyz", default = 'xyz')
parser.add_argument("--with_gradient_rotations", action="store_true", help="with learned gradient rotations")
parser.add_argument("--experiment", type=str, help="experiment name")
args = parser.parse_args()



# computing devices
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.gpu}')



# task settings
n_class = 21
class_names = "mIoU,none,\
wall,floor,cabinet,bed,chair,\
sofa,table,door,window,bookshelf,\
picture,counter,desk,curtain,refridgerator,\
shower curtain,toilet,sink,bathtub,otherfurniture\n"

# model settings
input_features = args.input_features # one of ['xyz', 'xyzrgb, 'hks']
k_eig = 128



# training settings
train = not args.evaluate
n_epoch = 25
lr = 1e-3
augment_random_rotate = (input_features == 'xyz') | (input_features == 'xyzrgb')
with_rgb = (input_features == 'xyzrgb')
with_gradient_rotations = args.with_gradient_rotations


# paths
experiment = args.experiment
repo_dir = "/home/cychen/Documents/GDL-scene-segment/ScanNet"
data_dir = "/media/cychen/HDD/scannet"
# repo_dir = "/home/chihyu/GDL-scene-segment/ScanNet"
# data_dir = "/shared/scannet"
op_cache_dir = Path(data_dir, "diffusion-net", f"op_cache_{k_eig}")
op_cache_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(repo_dir, "..", "pretrained_models", experiment)
model_dir.mkdir(parents=True, exist_ok=True)
pretrain_path = Path(model_dir, f"scannet_semseg_{input_features}.pth")
model_save_path = pretrain_path
pred_dir = Path(data_dir, "preds", experiment)
pred_dir.mkdir(parents=True, exist_ok=True)



# datasets
test_dataset = ScanNetDataset(train=False, repo_dir=repo_dir, data_dir=data_dir, with_rgb=with_rgb, k_eig=k_eig, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

if train:
    train_dataset = ScanNetDataset(train=True, repo_dir=repo_dir, data_dir=data_dir, with_rgb=with_rgb, k_eig=k_eig, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# # precompute operators and store in op_cache_dir
# for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene in tqdm(test_loader):
#     pass
# for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene in tqdm(train_loader):
#     pass



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
    print(f"Loading pretrained model from: {pretrain_path}")
    model.load_state_dict(torch.load(str(pretrain_path)))



# the optimizer & learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# DiffusionNet human segmentation
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.5, verbose=True)
# PicassoNet++ 
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, verbose=True)
# VMNet & DGNet
# lr_lambda = lambda epoch: (1 - epoch/n_epoch) ** 0.9
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)



# the training epoch
def train_epoch():

    # set model to 'train' mode
    model.train()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    for verts, rgb, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, _ in tqdm(train_loader):

        optimizer.zero_grad()

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
        
        # randomly rotate positions
        if augment_random_rotate:
            verts = utils.random_rotate_points_z(verts)

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

        # evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        total_loss += loss.item()
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class, device)
        tps += this_tps.cpu()
        fps += this_fps.cpu()
        fns += this_fns.cpu()

        # step the optimizer
        optimizer.step()

    ious = tps / (tps+fps+fns)

    return total_loss, np.insert(ious, 0, ious[1:].mean())



# the validation or test
def test(save=False):
    
    model.eval()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    with torch.no_grad():
    
        for verts, rgb, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene in tqdm(test_loader):

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
            pred_labels = torch.max(preds, dim=1).indices
            this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class, device)
            tps += this_tps.cpu()
            fps += this_fps.cpu()
            fns += this_fns.cpu()

            # save prediction
            if save:
                pred_labels = test_dataset.classes[pred_labels.cpu()]
                np.savetxt(pred_dir/f"{scene}_labels.txt", pred_labels, fmt='%d', delimiter='\n')

    ious = tps / (tps+fps+fns)

    return total_loss, np.insert(ious, 0, ious[1:].mean())



# actual running
torch.set_printoptions(sci_mode=False)
filestem = model_save_path.parent/model_save_path.stem

if train:

    print("Training...")

    with open(str(filestem)+"_train_ious.csv", 'w') as f:
        f.write(class_names)
    with open(str(filestem)+"_test_ious.csv", 'w') as f:
        f.write(class_names)

    for epoch in range(n_epoch):

        train_loss, train_ious = train_epoch()
        test_loss, test_ious = test()
        scheduler.step()

        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_ious[0]}\n")
        print(f"Test Loss: {test_loss:.4f}, Test mIoU: {test_ious[0]}")

        with open(str(filestem)+"_train_ious.csv", 'ab') as f:
            np.savetxt(f, train_ious[np.newaxis,:], delimiter=',')
        with open(str(filestem)+"_test_ious.csv", 'ab') as f:
            np.savetxt(f, test_ious[np.newaxis,:], delimiter=',')
        with open(str(filestem)+"_loss.csv", 'a') as f:
            f.write(str(train_loss)+",")
            f.write(str(test_loss)+"\n")

    torch.save(model.state_dict(), str(model_save_path))
    print(f" ==> saving last model to {model_save_path}")

test_loss, test_ious = test(save=True)
print(f"Overall Test Loss: {test_loss:.4f}, Test mIoU: {test_ious[0]}")
