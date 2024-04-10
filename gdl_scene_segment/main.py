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
from datasets.scannet_dataset import ScanNetDataset
from datasets.compute_operators import compute_gradients
from model import utils
from config.config import settings

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

c_in = {'xyz':3, 'xyzrgb': 6, 'hks':16, 'hksrgb': 19}[input_features]
c_out = n_class
c0 = settings.model.c0
mlp_hidden_dims = [c0] * n_mlp_hidden
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
random_offset = settings.training.augment.translate
random_flip = settings.training.augment.flip
random_scale = settings.training.augment.scale
chromatic_jitter = settings.training.augment.rgb_jitter
augment_operators = settings.training.augment.operators
translate_scale = settings.training.augment.translate_scale
scaling_range = settings.training.augment.scaling_range



# w&b setup
wandb.init(
    project="gdl_scene_segment",
    name=experiment,
    config=settings.to_dict()
)



# datasets
if train:
    train_dataset = ScanNetDataset(train=True, test=False,
                                   data_dir=data_dir,
                                   preprocess=preprocess,
                                   k_eig=k_eig,
                                   op_cache_dir=op_cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

val_dataset = ScanNetDataset(train=False, test=False,
                             data_dir=data_dir,
                             preprocess=preprocess,
                             k_eig=k_eig,
                             op_cache_dir=op_cache_dir)
val_loader = DataLoader(val_dataset, batch_size=None)

# the model
model = diffusion_net.layers.DiffusionNet(C_in=c_in,
                                          C_out=n_class,
                                          C_width=c0,
                                          N_block=n_diffnet_blocks,
                                          mlp_hidden_dims=mlp_hidden_dims,
                                          outputs_at='vertices',
                                          dropout=dropout,
                                          with_gradient_rotations=gradient_rotation,
)

model = model.to(device)
num_params = 0
for names, params in model.named_parameters():
    if params.requires_grad:
        print(names)
        num_params += params.numel()
print(f"number of parameters: {num_params}")



# load the pretrained model
if not train:
    print(f"Loading pretrained model from: {model_path}")
    model.load_state_dict(torch.load(model_path.as_posix()))



# the optimizer & learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

lr_scheduler = settings.training.lr_scheduler
if lr_scheduler == "step":
    # DiffusionNet & PicassoNet++
    lr_step_size = settings.training.lr_step_size
    gamma = settings.training.lr_step_gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma, verbose=True)
elif lr_scheduler == "lambda":
    # VMNet & DGNet
    lr_lambda = lambda epoch: (1 - epoch/(n_epoch+1)) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)



# the training epoch
def train_epoch():

    model.train()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    optimizer.zero_grad()

    for i, (_, verts, faces, rgb, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, norm_max) in enumerate(tqdm(train_loader)):

        # augmentation
        if random_rotate:
            rot_mat = utils.random_rotate_points_z()
            verts = torch.matmul(verts, rot_mat)

        if random_offset:
            offset = utils.random_translate(scale=translate_scale)
            verts += offset
        
        if random_flip:
            sign = utils.random_flip()
            verts[:,0] *= sign

        if random_scale:
            scale = utils.random_scale(scaling_range=scaling_range)
            verts *= scale
        
        if augment_operators:
            gradX, gradY = compute_gradients(verts/norm_max, faces, L)

        # normalize
        verts = verts / norm_max

        # rgb features
        if chromatic_jitter:
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
            x_in = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        elif input_features == 'hksrgb':
            hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            x_in = torch.hstack((hks, rgb)).float()

        # move to device
        x_in = x_in.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # apply the model
        out = model(x_in, mass, L, evals, evecs, gradX, gradY)

        # evaluate loss
        loss = loss_f(out, labels)
        total_loss += loss.item()
        loss.backward()
        
        # track accuracy
        preds = torch.argmax(out.cpu(), dim=-1)
        preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                 ).put_(ref_idx, preds)
        labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                  ).put_(ref_idx, labels.cpu())
        this_tps, this_fps, this_fns = utils.get_ious(preds, labels, n_class)
        tps += this_tps
        fps += this_fps
        fns += this_fns

        # step the optimizer
        if (i+1) % pseudo_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    ious = tps / (tps+fps+fns)

    return total_loss/len(train_loader), np.insert(ious, 0, ious.mean())



# the validation or test
def val(save_pred=False):
    
    model.eval()

    total_loss = 0
    tps = torch.zeros(n_class)
    fps = torch.zeros(n_class)
    fns = torch.zeros(n_class)

    with torch.no_grad():
    
        for scene, verts, _, rgb, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, norm_max in tqdm(val_loader):

            # normalize
            verts = verts / norm_max

            # construct features
            if input_features == 'xyz':
                x_in = verts.float()
            elif input_features == 'xyzrgb':
                x_in = torch.hstack((verts, rgb)).float()
            elif input_features == 'hks':
                x_in = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            elif input_features == 'hksrgb':
                hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
                x_in = torch.hstack((hks, rgb)).float()

            # move to device
            x_in = x_in.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # apply the model
            out = model(x_in, mass, L, evals, evecs, gradX, gradY)

            # track loss
            loss = loss_f(out, labels)
            total_loss += loss.item()
            
            # track accuracy
            preds = torch.argmax(out.cpu(), dim=-1)
            preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                    ).put_(ref_idx, preds)
            labels = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                    ).put_(ref_idx, labels.cpu())
            this_tps, this_fps, this_fns = utils.get_ious(preds, labels, n_class)
            tps += this_tps
            fps += this_fps
            fns += this_fns

            # save prediction
            if save_pred:
                preds[preds == -100] = -1
                val_dataset.classes = np.append(val_dataset.classes, [0])
                preds = val_dataset.classes[preds]
                np.savetxt(pred_dir/ f"{scene}.txt", preds, fmt='%d')
            
    ious = tps / (tps+fps+fns)

    return total_loss/len(val_loader), np.insert(ious, 0, ious.mean())



def test():

    test_dataset = ScanNetDataset(train=False, test=True,
                                  data_dir=data_dir,
                                  preprocess=preprocess,
                                  k_eig=k_eig,
                                  op_cache_dir=op_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=None)

    model.eval()

    with torch.no_grad():
    
        for scene, verts, _, rgb, mass, L, evals, evecs, gradX, gradY, _, ref_idx, norm_max in tqdm(test_loader):

            # normalize
            verts = verts / norm_max

            # construct features
            if input_features == 'xyz':
                x_in = verts.float()
            elif input_features == 'xyzrgb':
                x_in = torch.hstack((verts, rgb)).float()
            elif input_features == 'hks':
                x_in = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            elif input_features == 'hksrgb':
                hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
                x_in = torch.hstack((hks, rgb)).float()

            # move to device
            x_in = x_in.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            
            # apply the model
            out = model(x_in, mass, L, evals, evecs, gradX, gradY)
            
            # save prediction
            preds = torch.argmax(out.cpu(), dim=-1)
            preds = (-100 * torch.ones(ref_idx.max()+1, dtype=torch.int64)
                    ).put_(ref_idx, preds)
            preds[preds == -100] = -1
            test_dataset.classes = np.append(test_dataset.classes, [0])
            preds = test_dataset.classes[preds]
            np.savetxt(pred_dir/ f"{scene}.txt", preds, fmt='%d')

    return



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
            torch.save(model.state_dict(), model_path.with_stem(f"checkpoint{epoch+1:03d}"))
            print(" ==> model checkpoint saved")

    torch.save(model.state_dict(), model_path)
    print(" ==> last model saved")

val_loss, val_ious = val(save_pred=True)
print(f"Last Val Loss: {val_loss:.4f}, Val mIoU: {val_ious[0]:.4f}")
test()
wandb.finish()
