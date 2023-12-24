import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from torchsparse.utils.quantize import sparse_quantize
from tqdm import tqdm

import sys
pkg_path = Path(__file__).parents[1]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net
from datasets.scannet_hierarchy_dataset import ScanNetHierarchyDataset
from model import model, utils
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



# parse arguments outside python
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="directory to the ScanNet dataset")
parser.add_argument("--preprocess", type=str,
                    help="which preprocessing", required=True)
parser.add_argument("--gpu", type=str, default="0",
                    help="which gpu")
parser.add_argument("--evaluate", action="store_true",
                    help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, default = 'xyz',
                    help="'xyz', 'xyzrgb', or 'hks', default: xyz")
parser.add_argument("--experiment", type=str, required=True,
                    help="experiment name")
args = parser.parse_args()



# computing devices
device = torch.device(f'cuda:{args.gpu}')
torch.cuda.set_device(int(args.gpu))



# task settings
n_class = 20
class_names = "mIoU,\
wall,floor,cabinet,bed,chair,\
sofa,table,door,window,bookshelf,\
picture,counter,desk,curtain,refridgerator,\
shower curtain,toilet,sink,bathtub,otherfurniture\n"



# model settings
input_features = args.input_features # one of ['xyz', 'xyzrgb, 'hks']
n_levels = 4
with_rgb = ('rgb' in input_features)
k_eig = 128
n_diffnet_blocks = 2
n_mlp_hidden = 2
dropout = True
c_in = {'xyz':3, 'xyzrgb': 6, 'hks':16}[input_features]
c_out = n_class
c0 = 32
c1 = 64
c2 = 96
c3 = 128
c_m = 160
loss_f = torch.nn.functional.cross_entropy



# training settings
train = not args.evaluate
n_epoch = 200
pseudo_batch_size = 8
lr = 1e-3
lr_step_size = 50
checkpt_every = 10



# augmentation settings
augment_random_rotate = (input_features != 'hks')
translate_scale = 0.2
scaling_range = 0.5




# paths
experiment = args.experiment
data_dir = Path(args.data_dir)
preprocess = args.preprocess
op_cache_dir = data_dir/ "diffusion-net"/ f"op_cache_{k_eig}"
op_cache_dir.mkdir(parents=True, exist_ok=True)
exp_dir = Path("..", "experiments", experiment).resolve()
exp_dir.mkdir(parents=True, exist_ok=True)
model_path = exp_dir/ "model.pt"
pred_dir = exp_dir/ "preds"
pred_dir.mkdir(parents=True, exist_ok=True)



# datasets
val_dataset = ScanNetHierarchyDataset(train=False,
                                       data_dir=data_dir,
                                       preprocess=preprocess,
                                       n_levels=n_levels,
                                       with_rgb=with_rgb,
                                       k_eig=k_eig,
                                       op_cache_dir=op_cache_dir)
val_loader = DataLoader(val_dataset, batch_size=None)

if train:
    train_dataset = ScanNetHierarchyDataset(train=True,
                                            data_dir=data_dir,
                                            preprocess=preprocess,
                                            n_levels=n_levels,
                                            with_rgb=with_rgb,
                                            k_eig=k_eig,
                                            op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# the model
model = model.DiffusionVoxelNet(n_diffnet_blocks=n_diffnet_blocks,
                                n_mlp_hidden=n_mlp_hidden,
                                dropout=dropout,
                                c_in=c_in,
                                c_out=c_out,
                                c0=c0,
                                c1=c1,
                                c2=c2,
                                c3=c3,
                                c_m=c_m)

model = model.to(device)



# load the pretrained model
if not train:
    print(f"Loading pretrained model from: {model_path}")
    model.load_state_dict(torch.load(str(model_path)))



# the optimizer & learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# # DiffusionNet human segmentation
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5, verbose=True)
# PicassoNet++ 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, verbose=True)
# # VMNet & DGNet
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

    for i, (_, verts, rgbs, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, traces) in enumerate(tqdm(train_loader)):

        # unpack lists
        verts_0, verts_1 = verts
        rgb_0, rgb_1 = rgbs

        # mass_0, mass_1, mass_2, mass_3, mass_m = mass
        # L_0, L_1, L_2, L_3, L_m = L
        # evals_0, evals_1, evals_2, evals_3, evals_m = evals
        # evecs_0, evecs_1, evecs_2, evecs_3, evecs_m = evecs
        # gradX_0, gradX_1, gradX_2, gradX_3, gradX_m = gradX
        # gradY_0, gradY_1, gradY_2, gradY_3, gradY_m = gradY
        mass_1, mass_2, mass_3, mass_m = mass
        L_1, L_2, L_3, L_m = L
        evals_1, evals_2, evals_3, evals_m = evals
        evecs_1, evecs_2, evecs_3, evecs_m = evecs
        gradX_1, gradX_2, gradX_3, gradX_m = gradX
        gradY_1, gradY_2, gradY_3, gradY_m = gradY

        labels_0, labels_1 = labels

        traces01, traces12, traces23, traces34 = traces

        # normalize
        norm_max = np.linalg.norm(verts_0, axis=-1).max()

        # augmentation
        rot_mat = utils.random_rotate_points_z()
        offset = utils.random_translate(scale=translate_scale)
        sign = utils.random_flip()
        scale = utils.random_scale(scaling_range=scaling_range)

        if augment_random_rotate:
            verts_0 = torch.matmul(verts_0, rot_mat)
            verts_1 = torch.matmul(verts_1, rot_mat)
        verts_0 += offset
        verts_0[:,0] *= sign
        verts_0 *= scale
        verts_1 += offset
        verts_1[:,0] *= sign
        verts_1 *= scale

        # sparse-voxelize vertices
        voxels = verts_0.detach().numpy()
        voxels = voxels - voxels.min(axis=0)
        voxels, vox_idx = sparse_quantize(voxels, voxel_size=0.02, return_index=True)
        voxels = torch.tensor(voxels, dtype=torch.int)
        labels_vox = torch.tensor(labels_0[vox_idx], dtype=torch.long)

        # normalize
        verts_0 = verts_0 / norm_max
        verts_1 = verts_1 / norm_max

        # rgb features
        rgb_shape = rgb_0.shape
        jitter = utils.random_rgb_jitter(rgb_shape, scale=0.05)
        rgb_0 += jitter
        rgb_0 = torch.clamp(rgb_0, min=0, max=1)

        jitter = scatter_mean(torch.tensor(jitter), traces[0], dim=-2)
        rgb_1 += jitter
        rgb_1 = torch.clamp(rgb_1, min=0, max=1)

        rgb_vox = torch.tensor(rgb_0[vox_idx], dtype=torch.float)

        # construct features
        if input_features == 'xyz':
            x_in = verts_1
        elif input_features == 'xyzrgb':
            x_in = torch.hstack((verts_1, rgb_1))
        elif input_features == 'hks':
            x_in = diffusion_net.geometry.compute_hks_autoscale(evals_0, evecs_0, 16)

        # move to device
        x_in = x_in.to(device)
        voxels = voxels.to(device)
        rgb_vox = rgb_vox.to(device)

        # mass_0 = mass_0.to(device)
        mass_1 = mass_1.to(device)
        mass_2 = mass_2.to(device)
        mass_3 = mass_3.to(device)
        mass_m = mass_m.to(device)

        # L_0 = L_0.to(device)
        L_1 = L_1.to(device)
        L_2 = L_2.to(device)
        L_3 = L_3.to(device)
        L_m = L_m.to(device)

        # evals_0 = evals_0.to(device)
        evals_1 = evals_1.to(device)
        evals_2 = evals_2.to(device)
        evals_3 = evals_3.to(device)
        evals_m = evals_m.to(device)

        # evecs_0 = evecs_0.to(device)
        evecs_1 = evecs_1.to(device)
        evecs_2 = evecs_2.to(device)
        evecs_3 = evecs_3.to(device)
        evecs_m = evecs_m.to(device)

        # gradX_0 = gradX_0.to(device)
        gradX_1 = gradX_1.to(device)
        gradX_2 = gradX_2.to(device)
        gradX_3 = gradX_3.to(device)
        gradX_m = gradX_m.to(device)

        # gradY_0 = gradY_0.to(device)
        gradY_1 = gradY_1.to(device)
        gradY_2 = gradY_2.to(device)
        gradY_3 = gradY_3.to(device)
        gradY_m = gradY_m.to(device)

        labels_0 = labels_0.to(device)
        labels_1 = labels_1.to(device)
        labels_vox = labels_vox.to(device)

        traces01 = traces01.to(device)
        traces12 = traces12.to(device)
        traces23 = traces23.to(device)
        traces34 = traces34.to(device)
        
        # apply the model
        euc_out, geo_out = model(
            x_in, voxels, rgb_vox,
            # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
            mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
            mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
            mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
            mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
            traces01, traces12, traces23, traces34
        )

        # evaluate loss
        loss = loss_f(euc_out, labels_vox) + loss_f(geo_out, labels_1)
        total_loss += loss.item()
        loss.backward()
        
        # track accuracy
        geo_preds = torch.argmax(geo_out.cpu(), dim=-1)
        geo_preds = geo_preds[traces01.cpu()]
        geo_preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, geo_preds)
        gt_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, labels_0.cpu())
        this_tps, this_fps, this_fns = utils.get_ious(geo_preds, gt_labels, n_class)
        tps += this_tps
        fps += this_fps
        fns += this_fns

        # step the optimizer
        if (i+1) % pseudo_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    ious = tps / (tps+fps+fns)

    return total_loss/len(train_loader), np.insert(ious, 0, ious.mean())



# the validation
def val(save_pred=False):
    
    model.eval()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    with torch.no_grad():
    
        for scene, verts, rgbs, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, traces in tqdm(val_loader):

            # unpack lists
            verts_0, verts_1 = verts
            rgb_0, rgb_1 = rgbs

            # mass_0, mass_1, mass_2, mass_3, mass_m = mass
            # L_0, L_1, L_2, L_3, L_m = L
            # evals_0, evals_1, evals_2, evals_3, evals_m = evals
            # evecs_0, evecs_1, evecs_2, evecs_3, evecs_m = evecs
            # gradX_0, gradX_1, gradX_2, gradX_3, gradX_m = gradX
            # gradY_0, gradY_1, gradY_2, gradY_3, gradY_m = gradY
            mass_1, mass_2, mass_3, mass_m = mass
            L_1, L_2, L_3, L_m = L
            evals_1, evals_2, evals_3, evals_m = evals
            evecs_1, evecs_2, evecs_3, evecs_m = evecs
            gradX_1, gradX_2, gradX_3, gradX_m = gradX
            gradY_1, gradY_2, gradY_3, gradY_m = gradY

            labels_0, labels_1 = labels

            traces01, traces12, traces23, traces34 = traces

            # normalize
            norm_max = np.linalg.norm(verts_0, axis=-1).max()

            # sparse-voxelize vertices
            voxels = verts_0.detach().numpy()
            voxels = voxels - voxels.min(axis=0)
            voxels, vox_idx = sparse_quantize(voxels, voxel_size=0.02, return_index=True)
            voxels = torch.tensor(voxels, dtype=torch.int)
            labels_vox = torch.tensor(labels_0[vox_idx], dtype=torch.long)

            # normalize
            verts_0 = verts_0 / norm_max
            verts_1 = verts_1 / norm_max
            
            # rgb features
            rgb_vox = torch.tensor(rgb_0[vox_idx], dtype=torch.float)
    
            # construct features
            if input_features == 'xyz':
                x_in = verts_1
            elif input_features == 'xyzrgb':
                x_in = torch.hstack((verts_1, rgb_1))
            elif input_features == 'hks':
                x_in = diffusion_net.geometry.compute_hks_autoscale(evals_0, evecs_0, 16)

            # move to device
            x_in = x_in.to(device)
            voxels = voxels.to(device)
            rgb_vox = rgb_vox.to(device)

            # mass_0 = mass_0.to(device)
            mass_1 = mass_1.to(device)
            mass_2 = mass_2.to(device)
            mass_3 = mass_3.to(device)
            mass_m = mass_m.to(device)

            # L_0 = L_0.to(device)
            L_1 = L_1.to(device)
            L_2 = L_2.to(device)
            L_3 = L_3.to(device)
            L_m = L_m.to(device)

            # evals_0 = evals_0.to(device)
            evals_1 = evals_1.to(device)
            evals_2 = evals_2.to(device)
            evals_3 = evals_3.to(device)
            evals_m = evals_m.to(device)

            # evecs_0 = evecs_0.to(device)
            evecs_1 = evecs_1.to(device)
            evecs_2 = evecs_2.to(device)
            evecs_3 = evecs_3.to(device)
            evecs_m = evecs_m.to(device)

            # gradX_0 = gradX_0.to(device)
            gradX_1 = gradX_1.to(device)
            gradX_2 = gradX_2.to(device)
            gradX_3 = gradX_3.to(device)
            gradX_m = gradX_m.to(device)

            # gradY_0 = gradY_0.to(device)
            gradY_1 = gradY_1.to(device)
            gradY_2 = gradY_2.to(device)
            gradY_3 = gradY_3.to(device)
            gradY_m = gradY_m.to(device)

            labels_0 = labels_0.to(device)
            labels_1 = labels_1.to(device)
            labels_vox = labels_vox.to(device)

            traces01 = traces01.to(device)
            traces12 = traces12.to(device)
            traces23 = traces23.to(device)
            traces34 = traces34.to(device)

            # apply the model
            euc_out, geo_out = model(
                x_in, voxels, rgb_vox,
                # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
                mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                traces01, traces12, traces23, traces34
            )

            # track loss
            loss = loss_f(euc_out, labels_vox) + loss_f(geo_out, labels_1)
            total_loss += loss.item()

            # track accuracy
            geo_preds = torch.argmax(geo_out.cpu(), dim=-1)
            geo_preds = geo_preds[traces01.cpu()]
            geo_preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, geo_preds)
            gt_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)).put_(ref_idx, labels_0.cpu())
            this_tps, this_fps, this_fns = utils.get_ious(geo_preds, gt_labels, n_class)
            tps += this_tps
            fps += this_fps
            fns += this_fns

            # save prediction
            if save_pred:
                geo_preds[geo_preds == -100] = -1
                val_dataset.classes = np.append(val_dataset.classes, [0])
                geo_preds = val_dataset.classes[geo_preds]
                np.savetxt(pred_dir/ f"{scene}_labels.txt", geo_preds, fmt='%d')
            
    ious = tps / (tps+fps+fns)

    return total_loss/len(val_loader), np.insert(ious, 0, ious.mean())



# actual running
torch.set_printoptions(sci_mode=False)

if train:

    print("Training...")

    with open(model_path.with_name("train_iou.csv"), 'w') as f:
        f.write(class_names)
    with open(model_path.with_name("val_iou.csv"), 'w') as f:
        f.write(class_names)

    for epoch in range(n_epoch):

        train_loss, train_ious = train_epoch()
        val_loss, val_ious = val()
        scheduler.step()

        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_ious[0]}")
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_ious[0]}")

        with open(model_path.with_name("train_iou.csv"), 'ab') as f:
            np.savetxt(f, train_ious[np.newaxis,:], delimiter=',')
        with open(model_path.with_name("val_iou.csv"), 'ab') as f:
            np.savetxt(f, val_ious[np.newaxis,:], delimiter=',')
        with open(model_path.with_name("loss.csv"), 'a') as f:
            f.write(str(train_loss)+",")
            f.write(str(val_loss)+"\n")
        
        if (epoch+1) % checkpt_every == 0:
            torch.save(model.state_dict(), model_path.with_stem(f"checkpoint{epoch+1}"))
            print(" ==> model checkpoint saved")

    torch.save(model.state_dict(), model_path)
    print(" ==> last model saved")

val_loss, val_ious = val(save_pred=True)
print(f"Overall Val Loss: {val_loss:.4f}, Val mIoU: {val_ious[0]}")
