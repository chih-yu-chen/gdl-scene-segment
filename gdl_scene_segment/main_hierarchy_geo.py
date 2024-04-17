import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys
pkg_path = Path(__file__).parents[1]/ "diffusion-net"/ "src"
sys.path.append(pkg_path.as_posix())
import diffusion_net

from datasets.scannet_hierarchy_dataset_test_geo import ScanNetHierarchyDataset
from model import model_test_geo, utils
from config.config import settings

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
wandb.login()



# parse arguments outside python
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="directory to the ScanNet dataset")
parser.add_argument("--gpu", type=str, default="0",
                    help="which gpu")
parser.add_argument("--evaluate", action="store_true",
                    help="evaluate using the pretrained model")
args = parser.parse_args()



# computing devices
device = torch.device(f'cuda:{args.gpu}')
torch.cuda.set_device(int(args.gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# paths
data_dir = Path(args.data_dir)
preprocess = settings.data.preprocess
experiment = settings.experiment.name
exp_dir = Path(__file__).parents[1]/ "experiments"/ experiment
exp_dir.mkdir(parents=True, exist_ok=True)
model_path = exp_dir/ "model.pt"
pred_dir = exp_dir/ "preds"
pred_dir.mkdir(parents=True, exist_ok=True)



# task settings
n_class = settings.data.n_class
class_names = settings.data.class_names



# model settings
input_features = settings.model.input_features
k_eig = settings.model.k_eig
op_cache_dir = data_dir/ "diffusion-net"/ f"op_cache_{k_eig}"
n_diffnet_blocks = settings.model.n_diffnet_blocks
n_mlp_hidden = settings.model.n_mlp_hidden
dropout = settings.model.dropout
gradient_rotation = settings.model.gradient_rotation

c_in = {'xyz':3, 'xyzrgb': 6, 'hks':16}[input_features]
c_out = n_class

n_levels = settings.model.hierarchy.n_levels
c1 = settings.model.hierarchy.c1
c2 = settings.model.hierarchy.c2
c3 = settings.model.hierarchy.c3
c_m = settings.model.hierarchy.c_m
loss_f = torch.nn.functional.cross_entropy



# training settings
train = not args.evaluate
n_epoch = settings.training.n_epoch
pseudo_batch_size = settings.training.pseudo_batch_size
lr = settings.training.learning_rate
wd = settings.training.weight_decay
checkpt_every = settings.training.checkpt_every



# augmentation settings
random_rotate = settings.training.augment.rotate
other_augment = settings.training.augment.other
translate_scale = settings.training.augment.translate_scale
scaling_range = settings.training.augment.scaling_range



# w&b setup
wandb.init(
    project="gdl_scene_segment",
    name=experiment,
    config=settings.to_dict()
)



# datasets
val_dataset = ScanNetHierarchyDataset(train=False,
                                      data_dir=data_dir,
                                      preprocess=preprocess,
                                      n_levels=n_levels,
                                      k_eig=k_eig,
                                      op_cache_dir=op_cache_dir)
val_loader = DataLoader(val_dataset, batch_size=None)

if train:
    train_dataset = ScanNetHierarchyDataset(train=True,
                                            data_dir=data_dir,
                                            preprocess=preprocess,
                                            n_levels=n_levels,
                                            k_eig=k_eig,
                                            op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# the model
m = model_test_geo.DiffusionVoxelNet(n_diffnet_blocks=n_diffnet_blocks,
                                     n_mlp_hidden=n_mlp_hidden,
                                     dropout=dropout,
                                     with_gradient_rotations=gradient_rotation,
                                     c_in=c_in,
                                     c_out=c_out,
                                     c3=c3,
                                     c_m=c_m
)

m = m.to(device)
num_params = 0
for names, params in m.named_parameters():
    if params.requires_grad:
        print(names)
        num_params += params.numel()
print(f"number of parameters: {num_params}")



# load the pretrained model
if not train:
    print(f"Loading pretrained model from: {model_path}")
    m.load_state_dict(torch.load(model_path.as_posix()))



# the optimizer & learning rate scheduler
optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
lr_step_size = settings.training.lr_step_size
gamma = settings.training.lr_step_gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma, verbose=True)
# # VMNet & DGNet
# lr_lambda = lambda epoch: (1 - epoch/(n_epoch+1)) ** 0.9
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)



# the training epoch
def train_epoch():

    m.train()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    optimizer.zero_grad()

    for i, (_, verts, _, rgb, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, norm_max, traces) in enumerate(tqdm(train_loader)):

        # unpack lists
        mass_3, mass_m = mass
        L_3, L_m = L
        evals_3, evals_m = evals
        evecs_3, evecs_m = evecs
        gradX_3, gradX_m = gradX
        gradY_3, gradY_m = gradY

        labels_0, labels_1 = labels

        traces01, traces12 = traces

        # augmentation
        if random_rotate:
            rot_mat = utils.random_rotate_points_z()
            verts = torch.matmul(verts, rot_mat)

        if other_augment:
            offset = utils.random_translate(scale=translate_scale)
            verts += offset
            sign = utils.random_flip()
            verts[:,0] *= sign
            scale = utils.random_scale(scaling_range=scaling_range)
            verts *= scale

        # normalize
        verts = verts / norm_max

        # rgb features
        if other_augment:
            rgb_shape = rgb.shape
            jitter = utils.random_rgb_jitter(rgb_shape, scale=0.05)
            rgb += jitter
            rgb = torch.clamp(rgb, min=0, max=1)

        # construct features
        if input_features == 'xyz':
            x_in = verts.float()
        elif input_features == 'xyzrgb':
            x_in = torch.hstack((verts, rgb)).float()
        elif input_features == 'hks':
            x_in = diffusion_net.geometry.compute_hks_autoscale(evals_3, evecs_3, 16)

        # move to device
        x_in = x_in.to(device)

        # mass_0 = mass_0.to(device)
        # mass_1 = mass_1.to(device)
        # mass_2 = mass_2.to(device)
        mass_3 = mass_3.to(device)
        mass_m = mass_m.to(device)

        # L_0 = L_0.to(device)
        # L_1 = L_1.to(device)
        # L_2 = L_2.to(device)
        L_3 = L_3.to(device)
        L_m = L_m.to(device)

        # evals_0 = evals_0.to(device)
        # evals_1 = evals_1.to(device)
        # evals_2 = evals_2.to(device)
        evals_3 = evals_3.to(device)
        evals_m = evals_m.to(device)

        # evecs_0 = evecs_0.to(device)
        # evecs_1 = evecs_1.to(device)
        # evecs_2 = evecs_2.to(device)
        evecs_3 = evecs_3.to(device)
        evecs_m = evecs_m.to(device)

        # gradX_0 = gradX_0.to(device)
        # gradX_1 = gradX_1.to(device)
        # gradX_2 = gradX_2.to(device)
        gradX_3 = gradX_3.to(device)
        gradX_m = gradX_m.to(device)

        # gradY_0 = gradY_0.to(device)
        # gradY_1 = gradY_1.to(device)
        # gradY_2 = gradY_2.to(device)
        gradY_3 = gradY_3.to(device)
        gradY_m = gradY_m.to(device)

        labels_0 = labels_0.to(device)
        labels_1 = labels_1.to(device)

        traces01 = traces01.to(device)
        traces12 = traces12.to(device)
        # traces23 = traces23.to(device)
        # traces34 = traces34.to(device)
        
        # apply the model
        geo_out = m(
            x_in,
            # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
            # mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
            # mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
            mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
            mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
            traces12,
            # traces23,
            # traces34
        )

        # evaluate loss
        loss = loss_f(geo_out, labels_1)
        total_loss += loss.item()
        loss.backward()
        
        # track accuracy
        geo_preds = torch.argmax(geo_out.cpu(), dim=-1)
        geo_preds = geo_preds[traces01.cpu()]
        geo_preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                     ).put_(ref_idx, geo_preds)
        gt_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                     ).put_(ref_idx, labels_0.cpu())
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
    
    m.eval()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    with torch.no_grad():
    
        for scene, verts, _, rgb, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, norm_max, traces in tqdm(val_loader):

            # unpack lists
            mass_3, mass_m = mass
            L_3, L_m = L
            evals_3, evals_m = evals
            evecs_3, evecs_m = evecs
            gradX_3, gradX_m = gradX
            gradY_3, gradY_m = gradY

            labels_0, labels_1 = labels

            traces01, traces12 = traces

            # normalize
            verts = verts / norm_max

            # construct features
            if input_features == 'xyz':
                x_in = verts.float()
            elif input_features == 'xyzrgb':
                x_in = torch.hstack((verts, rgb)).float()
            elif input_features == 'hks':
                x_in = diffusion_net.geometry.compute_hks_autoscale(evals_3, evecs_3, 16)

            # move to device
            x_in = x_in.to(device)

            # mass_0 = mass_0.to(device)
            # mass_1 = mass_1.to(device)
            # mass_2 = mass_2.to(device)
            mass_3 = mass_3.to(device)
            mass_m = mass_m.to(device)

            # L_0 = L_0.to(device)
            # L_1 = L_1.to(device)
            # L_2 = L_2.to(device)
            L_3 = L_3.to(device)
            L_m = L_m.to(device)

            # evals_0 = evals_0.to(device)
            # evals_1 = evals_1.to(device)
            # evals_2 = evals_2.to(device)
            evals_3 = evals_3.to(device)
            evals_m = evals_m.to(device)

            # evecs_0 = evecs_0.to(device)
            # evecs_1 = evecs_1.to(device)
            # evecs_2 = evecs_2.to(device)
            evecs_3 = evecs_3.to(device)
            evecs_m = evecs_m.to(device)

            # gradX_0 = gradX_0.to(device)
            # gradX_1 = gradX_1.to(device)
            # gradX_2 = gradX_2.to(device)
            gradX_3 = gradX_3.to(device)
            gradX_m = gradX_m.to(device)

            # gradY_0 = gradY_0.to(device)
            # gradY_1 = gradY_1.to(device)
            # gradY_2 = gradY_2.to(device)
            gradY_3 = gradY_3.to(device)
            gradY_m = gradY_m.to(device)

            labels_0 = labels_0.to(device)
            labels_1 = labels_1.to(device)

            traces01 = traces01.to(device)
            traces12 = traces12.to(device)
            # traces23 = traces23.to(device)
            # traces34 = traces34.to(device)

            # apply the model
            geo_out = m(
                x_in,
                # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
                # mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                # mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                traces12,
                # traces23,
                # traces34
            )

            # track loss
            loss = loss_f(geo_out, labels_1)
            total_loss += loss.item()

            # track accuracy
            geo_preds = torch.argmax(geo_out.cpu(), dim=-1)
            geo_preds = geo_preds[traces01.cpu()]
            geo_preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                         ).put_(ref_idx, geo_preds)
            gt_labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                         ).put_(ref_idx, labels_0.cpu())
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
        f.write(class_names+"\n")
    with open(model_path.with_name("val_iou.csv"), 'w') as f:
        f.write(class_names+"\n")
    with open(model_path.with_name("metrics.csv"), 'w') as f:
        f.write("Train_Loss,Val_Loss,Train_mIoU,Val_mIoU\n")

    for epoch in range(n_epoch):

        train_loss, train_ious = train_epoch()
        val_loss, val_ious = val()
        scheduler.step()

        wandb.log({
            "train/loss": train_loss,
            "train/mIoU": train_ious[0],
            "val/loss": val_loss,
            "val/mIoU": val_ious[0],
        })
        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_ious[0]:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_ious[0]:.4f}")

        with open(model_path.with_name("train_iou.csv"), 'ab') as f:
            np.savetxt(f, train_ious[np.newaxis,:], delimiter=',', fmt='%.4f')
        with open(model_path.with_name("val_iou.csv"), 'ab') as f:
            np.savetxt(f, val_ious[np.newaxis,:], delimiter=',', fmt='%.4f')
        metrics = np.asarray([train_loss, val_loss, train_ious[0], val_ious[0]])
        with open(model_path.with_name("metrics.csv"), 'ab') as f:
            np.savetxt(f, metrics[np.newaxis,:], delimiter=',', fmt='%.4f')
        
        if (epoch+1) % checkpt_every == 0:
            torch.save(m.state_dict(), model_path.with_stem(f"checkpoint{epoch+1:03d}"))
            print(" ==> model checkpoint saved")

    torch.save(m.state_dict(), model_path)
    print(" ==> last model saved")

val_loss, val_ious = val(save_pred=True)
print(f"Last Val Loss: {val_loss:.4f}, Val mIoU: {val_ious[0]:.4f}")
wandb.finish()
