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
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')



# important paths
repo_dir = "/home/cychen/Documents/thesis/ScanNet"
data_dir = "/media/cychen/HDD/scannet"
op_cache_dir = Path(data_dir, "diffusion-net", "op_cache")
pretrain_path = Path(data_dir, "pretrained_models", f"scannet_semseg_{input_features}.pth")
model_save_path = Path(data_dir, "pretrained_models", f"scannet_semseg_{input_features}.pth")



# load the test dataset
test_dataset = ScanNetDataset(train=False, repo_dir=repo_dir, data_dir=data_dir, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

# # load the train dataset
# if train:
#     train_dataset = ScanNetDataset(train=True, repo_dir=repo_dir, data_dir=data_dir, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
#     train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# # create the model

# C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

# model = diffusion_net.layers.DiffusionNet(C_in=C_in,
#                                           C_out=n_class,
#                                           C_width=128, 
#                                           N_block=4, 
#                                           last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
#                                           outputs_at='faces', 
#                                           dropout=True)


# model = model.to(device)

# if not train:
#     # load the pretrained model
#     print("Loading pretrained model from: " + str(pretrain_path))
#     model.load_state_dict(torch.load(pretrain_path))


# # === Optimize
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# def train_epoch(epoch):

#     # Implement lr decay
#     if epoch > 0 and epoch % decay_every == 0:
#         global lr 
#         lr *= decay_rate
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr 


#     # Set model to 'train' mode
#     model.train()
#     optimizer.zero_grad()
    
#     correct = 0
#     total_num = 0
#     for data in tqdm(train_loader):

#         # Get data
#         verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

#         # Move to device
#         verts = verts.to(device)
#         faces = faces.to(device)
#         frames = frames.to(device)
#         mass = mass.to(device)
#         L = L.to(device)
#         evals = evals.to(device)
#         evecs = evecs.to(device)
#         gradX = gradX.to(device)
#         gradY = gradY.to(device)
#         labels = labels.to(device)
        
#         # Randomly rotate positions
#         if augment_random_rotate:
#             verts = diffusion_net.utils.random_rotate_points(verts)

#         # Construct features
#         if input_features == 'xyz':
#             features = verts
#         elif input_features == 'hks':
#             features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

#         # Apply the model
#         preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

#         # Evaluate loss
#         loss = torch.nn.functional.nll_loss(preds, labels)
#         loss.backward()
        
#         # track accuracy
#         pred_labels = torch.max(preds, dim=1).indices
#         this_correct = pred_labels.eq(labels).sum().item()
#         this_num = labels.shape[0]
#         correct += this_correct
#         total_num += this_num

#         # Step the optimizer
#         optimizer.step()
#         optimizer.zero_grad()

#     train_acc = correct / total_num
#     return train_acc


# # Do an evaluation pass on the test dataset 
# def test():
    
#     model.eval()
    
#     correct = 0
#     total_num = 0
#     with torch.no_grad():
    
#         for data in tqdm(test_loader):

#             # Get data
#             verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

#             # Move to device
#             verts = verts.to(device)
#             faces = faces.to(device)
#             frames = frames.to(device)
#             mass = mass.to(device)
#             L = L.to(device)
#             evals = evals.to(device)
#             evecs = evecs.to(device)
#             gradX = gradX.to(device)
#             gradY = gradY.to(device)
#             labels = labels.to(device)
            
#             # Construct features
#             if input_features == 'xyz':
#                 features = verts
#             elif input_features == 'hks':
#                 features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

#             # Apply the model
#             preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

#             # track accuracy
#             pred_labels = torch.max(preds, dim=1).indices
#             this_correct = pred_labels.eq(labels).sum().item()
#             this_num = labels.shape[0]
#             correct += this_correct
#             total_num += this_num

#     test_acc = correct / total_num
#     return test_acc 


# if train:
#     print("Training...")

#     for epoch in range(n_epoch):
#         train_acc = train_epoch(epoch)
#         test_acc = test()
#         print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))

#     print(" ==> saving last model to " + model_save_path)
#     torch.save(model.state_dict(), model_save_path)


# # Test
# test_acc = test()
# print("Overall test accuracy: {:06.3f}%".format(100*test_acc))







# # Here we use Nx3 positions as features. Any other features you might have will work!
# # See our experiments for the use of of HKS features, which are naturally 
# # invariant to (isometric) deformations.
# C_in = 3

# # Output dimension (e.g., for a 10-class segmentation problem)
# C_out = 10 

# # Create the model
# model = diffusion_net.layers.DiffusionNet(
#             C_in=C_in,
#             C_out=n_class,
#             C_width=128, # internal size of the diffusion net. 32 -- 512 is a reasonable range
#             last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1), # apply a last softmax to outputs 
#                                                                                   # (set to default None to output general values in R^{N x C_out})
#             outputs_at='vertices')

# # An example epoch loop.
# # For a dataloader example see experiments/human_segmentation_original/human_segmentation_original_dataset.py
# for sample in your_dataset:
    
#     verts = sample.vertices  # (Vx3 array of vertices)
#     faces = sample.faces     # (Fx3 array of faces, None for point cloud) 
    
#     # center and unit scale
#     verts = diffusion_net.geometry.normalize_positions(verts)
    
#     # Get the geometric operators needed to evaluate DiffusionNet. This routine 
#     # automatically populates a cache, precomputing only if needed.
#     # TIP: Do this once in a dataloader and store in memory to further improve 
#     # performance; see examples.
#     frames, mass, L, evals, evecs, gradX, gradY = \
#         get_operators(verts, faces, op_cache_dir='my/cache/directory/')
    
#     # this example uses vertex positions as features 
#     features = verts
    
#     # Forward-evaluate the model
#     # preds is a NxC_out array of values
#     outputs = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
    
#     # Now do whatever you want! Apply your favorite loss function, 
#     # backpropgate with loss.backward() to train the DiffusionNet, etc. 