import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d
from plyfile import PlyData

pkg_path = Path(__file__).parent/"diffusion-net"/"src"
sys.path.append(str(pkg_path))
import diffusion_net



class ScanNetDataset(Dataset):

    def __init__(self, train, repo_dir, data_dir, with_rgb=False, k_eig=128, op_cache_dir=None):

        self.train = train
        self.repo_dir = Path(repo_dir)
        self.data_dir = Path(data_dir)
        self.with_rgb = with_rgb
        self.k_eig = k_eig 
        self.op_cache_dir = Path(op_cache_dir)
        self.classes = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.label_map = np.ones(41, dtype=np.int8) * -100
        np.put(self.label_map, self.classes, np.arange(self.classes.size))

        # load train test split
        split_dir = self.repo_dir/"Tasks"/"Benchmark"
        if self.train:
            with open(split_dir/"scannetv2_train.txt", 'r') as f:
                self.scene_list = f.read().splitlines()
        else:
            with open(split_dir/"scannetv2_val.txt", 'r') as f:
                self.scene_list = f.read().splitlines()

    def __len__(self):

        return len(self.scene_list)

    def __getitem__(self, idx):

        scene = self.scene_list[idx]

        # get mesh and label paths
        # if self.train:
        train_dir = self.data_dir/"scans"
        mesh_path = train_dir/scene/(scene+"_vh_clean_2.ply")
        label_path = train_dir/scene/(scene+"_vh_clean_2.labels.ply")
        # else:
        #     test_dir = self.data_dir/"scans_test"
        #     mesh_path = test_dir/scene/(scene+"_vh_clean_2.ply")

        # load mesh
        verts, faces = pp3d.read_mesh(str(mesh_path))
        verts = torch.tensor(np.ascontiguousarray(verts)).float()
        faces = torch.tensor(np.ascontiguousarray(faces))

        # center and unit scale
        verts = diffusion_net.geometry.normalize_positions(verts)

        # load rgb
        rgb = None
        if self.with_rgb:
            rgb_path = self.data_dir/"rgb"/f"{scene}_rgb.txt"
            rgb = np.loadtxt(rgb_path, delimiter=',')
            rgb = torch.tensor(np.ascontiguousarray(rgb)).float()

        # precompute operators
        frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, self.k_eig, self.op_cache_dir)

        # if not self.train:
        #     return verts, faces, frames, massvec, L, evals, evecs, gradX, gradY

        # load labels
        with open(label_path, 'rb') as f:
            plydata = PlyData.read(f)
        labels = plydata['vertex'].data['label']
        labels = self.label_map[labels]
        labels = torch.tensor(np.ascontiguousarray(labels.astype(np.int64)))

        return verts, rgb, faces, frames, massvec, L, evals, evecs, gradX, gradY, labels, scene
