import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
pkg_path = Path(__file__).parent/"diffusion-net"/"src"
# import os
# os.getcwd()
# pkg_path = Path(os.getcwd())/"diffusion-net"/"src"
sys.path.append(str(pkg_path))
import diffusion_net
from scannet_dataset import ScanNetDataset
import utils



# parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()



# system things
device = torch.device('cuda:0')
dtype = torch.float32

# problem/dataset things
n_class = 21
class_names = "\
mIoU,none,\
wall,floor,cabinet,bed,chair,\
sofa,table,door,window,bookshelf,\
picture,counter,desk,curtain,refridgerator,\
shower curtain,toilet,sink,bathtub,otherfurniture"

# model 
input_features = args.input_features # one of ['xyz', 'hks']
# input_features = 'xyz'
k_eig = 128

# training settings
train = not args.evaluate
# train = False
n_epoch = 10
lr = 1e-3
decay_every = 5
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')


# important paths
# repo_dir = "/home/cychen/Documents/GDL-scene-segment/ScanNet"
# data_dir = "/media/cychen/HDD/scannet"
repo_dir = "/home/chihyu/GDL-scene-segment/ScanNet"
data_dir = "/shared/scannet"
op_cache_dir = Path(data_dir, "diffusion-net", "op_cache")
model_dir = Path(repo_dir, "..", "pretrained_models")
model_dir.mkdir(parents=True, exist_ok=True)
pretrain_path = Path(model_dir, f"scannet_semseg_{input_features}.pth")
model_save_path = pretrain_path
pred_dir = Path(data_dir)/"preds"
pred_dir.mkdir(parents=True, exist_ok=True)



# load the test dataset
test_dataset = ScanNetDataset(train=False, repo_dir=repo_dir, data_dir=data_dir, k_eig=k_eig, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# load the train dataset
if train:
    train_dataset = ScanNetDataset(train=True, repo_dir=repo_dir, data_dir=data_dir, k_eig=k_eig, op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# # precompute operators and store in op_cache_dir
# for verts, faces, frames, massvec, L, evals, evecs, gradX, gradY in tqdm(test_loader):
#     pass
# for verts, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels in tqdm(train_loader):
#     pass



# create the model
C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in, C_out=n_class,
                                          C_width=128, N_block=4,
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='vertices',
                                          dropout=True)

model = model.to(device)

if not train:
    # load the pretrained model
    print(f"Loading pretrained model from: {pretrain_path}")
    model.load_state_dict(torch.load(str(pretrain_path)))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)



def train_epoch(epoch):

    # implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 


    # set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    tps = torch.zeros(n_class).to(device)
    fps = torch.zeros(n_class).to(device)
    fns = torch.zeros(n_class).to(device)
    for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene in tqdm(train_loader):

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
            verts = diffusion_net.utils.random_rotate_points(verts)

        # construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # evaluate loss
        loss = torch.nn.functional.nll_loss(preds, labels)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=1).indices
        this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class, device)
        tps += this_tps
        fps += this_fps
        fns += this_fns

        # step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    ious = tps / (tps+fps+fns)

    return np.insert(ious.cpu(), 0, ious[1:].cpu().mean())



# do an evaluation pass on the test dataset 
def test(save=False):
    
    model.eval()
    
    tps = torch.zeros(n_class).to(device)
    fps = torch.zeros(n_class).to(device)
    fns = torch.zeros(n_class).to(device)
    with torch.no_grad():
    
        for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene in tqdm(test_loader):

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
            
            # construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=1).indices
            this_tps, this_fps, this_fns = utils.get_ious(pred_labels, labels, n_class, device)
            tps += this_tps
            fps += this_fps
            fns += this_fns

            if save:
                pred_labels = test_dataset.classes[pred_labels.cpu()]
                np.savetxt(pred_dir/f"{scene}_labels.txt", pred_labels, fmt='%d', delimiter='\n')

    ious = tps / (tps+fps+fns)

    return np.insert(ious.cpu(), 0, ious[1:].cpu().mean())



torch.set_printoptions(sci_mode=False)
train_ious_rec = []
test_ious_rec = []

if train:

    print("Training...")

    for epoch in range(n_epoch):
        train_ious = train_epoch(epoch)
        test_ious = test()
        train_ious_rec.append(train_ious)
        test_ious_rec.append(test_ious)
        print(f"Epoch {epoch}\nTrain IoU: {train_ious}\nTest IoU: {test_ious}")

    torch.save(model.state_dict(), str(model_save_path))
    print(f" ==> saving last model to {model_save_path}")
    ious_filename = model_save_path.parent/model_save_path.stem
    np.savetxt(str(ious_filename)+"train_ious.csv", np.vstack(train_ious_rec), delimiter=',', header=class_names, comments='')
    np.savetxt(str(ious_filename)+"test_ious.csv", np.vstack(test_ious_rec), delimiter=',', header=class_names, comments='')

test_ious = test(save=True)
print(f"Overall test IoU: {test_ious}")
