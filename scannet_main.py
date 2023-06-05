import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
pkg_path = Path(__file__).parent/"diffusion-net"/"src"
sys.path.append(str(pkg_path))
import diffusion_net
from scannet_dataset import ScanNetDataset

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

# model 
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 6
lr = 1e-3
decay_every = 2
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')



# important paths
repo_dir = "/home/cychen/Documents/GDL-scene-segment/ScanNet"
data_dir = "/media/cychen/HDD/scannet"
op_cache_dir = Path(data_dir, "diffusion-net", "op_cache")
pretrain_path = Path(repo_dir, "..", "pretrained_models", f"scannet_semseg_{input_features}.pth")
model_save_path = Path(repo_dir, "..", "pretrained_models", f"scannet_semseg_{input_features}.pth")



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
    
    correct = 0
    total_num = 0
    for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels in tqdm(train_loader):

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
        this_correct = pred_labels.eq(labels).sum().item()
        this_num = labels.shape[0]
        correct += this_correct
        total_num += this_num

        # step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num

    return train_acc


# do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels in tqdm(test_loader):

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
            this_correct = pred_labels.eq(labels).sum().item()
            this_num = labels.shape[0]
            correct += this_correct
            total_num += this_num

    test_acc = correct / total_num

    return test_acc 


if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        test_acc = test()
        print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

    torch.save(model.state_dict(), str(model_save_path))
    print(f" ==> saving last model to {model_save_path}")

test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
