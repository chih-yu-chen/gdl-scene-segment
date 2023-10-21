import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from scannet_dataset import ScanNetDataset



# parse arguments outside python
parser = argparse.ArgumentParser()
parser.add_argument("--machine", type=str, help="which machine", required=True)
parser.add_argument("--preprocess", type=str, help="which preprocessing", required=True)
args = parser.parse_args()



# model settings
k_eig = 128



# paths
if args.machine == "room":
    repo_dir = "/home/cychen/Documents/gdl-scene-segment/ScanNet"
    data_dir = "/media/cychen/HDD/scannet"
elif args.machine == "hal":
    repo_dir = "/home/chihyu/gdl-scene-segment/ScanNet"
    data_dir = "/shared/scannet"
op_cache_dir = Path(data_dir, "diffusion-net", f"op_cache_{k_eig}")
op_cache_dir.mkdir(parents=True, exist_ok=True)



# datasets
test_dataset = ScanNetDataset(train=False, repo_dir=repo_dir, data_dir=data_dir, with_rgb=False, preprocess=args.preprocess, k_eig=k_eig, op_cache_dir=op_cache_dir)
test_loader = DataLoader(test_dataset, batch_size=None)

train_dataset = ScanNetDataset(train=True, repo_dir=repo_dir, data_dir=data_dir, with_rgb=False, preprocess=args.preprocess, k_eig=k_eig, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)



# compute operators and store in op_cache_dir
for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene, ref_idx in tqdm(test_loader):
    pass
for verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, scene, ref_idx in tqdm(train_loader):
    pass
