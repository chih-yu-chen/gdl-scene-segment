import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(pkg_path.as_posix())
import diffusion_net



class ScanNetDataset(Dataset):

    def __init__(self,
                 train:bool,
                 test:bool,
                 data_dir:Path,
                 preprocess:str,
                 k_eig:int,
                 op_cache_dir:Path):

        self.train = train
        self.test = test
        self.data_dir = data_dir/ preprocess
        self.preprocess = preprocess
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir/ preprocess
        self.classes = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.label_map = np.ones(41, dtype=np.int8) * -100
        np.put(self.label_map, self.classes, np.arange(self.classes.size))

        # load train test split
        split_dir = Path(__file__).parent/ "splits"
        if self.train:
            with open(split_dir/ "scannetv2_train.txt", 'r') as f:
            # with open(split_dir/ "scannetv2_train_3e5.txt", 'r') as f:
                self.scene_list = f.read().splitlines()
        elif self.test:
            with open(split_dir/ "scannetv2_test.txt", 'r') as f:
                self.scene_list = f.read().splitlines()
        else:
            with open(split_dir/ "scannetv2_val.txt", 'r') as f:
                self.scene_list = f.read().splitlines()

        return

    def __len__(self):

        return len(self.scene_list)

    def __getitem__(self, idx):

        scene = self.scene_list[idx]

        # load mesh
        mesh_path = self.data_dir/ "scenes"/ f"{scene}_vh_clean_2.ply"
        verts, faces = pp3d.read_mesh(mesh_path.as_posix())
        verts = torch.tensor(np.ascontiguousarray(verts)).float()
        faces = torch.tensor(np.ascontiguousarray(faces.astype(np.int32)))

        # unit scale
        with open(self.data_dir/ "norm_max"/ f"{scene}_norm_max.txt", 'r') as f:
            norm_max = float(f.read())
        verts = verts / norm_max

        # load operators
        _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, self.k_eig, self.op_cache_dir)

        # scale back
        verts = verts * norm_max

        # load labels
        if self.test:
            labels = []
        else:
            label_path = self.data_dir/ "labels"/ f"{scene}_labels.txt"
            labels = np.loadtxt(label_path, dtype=np.int8)
            labels = self.label_map[labels]
            labels = torch.tensor(np.ascontiguousarray(labels.astype(np.int64)))

        # load rgb
        rgb_path = self.data_dir/ "rgb"/ f"{scene}_rgb.txt"
        rgb = np.loadtxt(rgb_path, delimiter=',', dtype=np.uint8)
        rgb = rgb / 255.
        rgb = torch.tensor(np.ascontiguousarray(rgb)).float()

        # load idx
        if self.preprocess == "centered":
            ref_idx = np.arange(verts.shape[0], dtype=np.int64)
        else:
            idx_path = self.data_dir/ "idx"/ f"{scene}_referenced_idx.txt"
            ref_idx = np.loadtxt(idx_path, dtype=np.int64)
        ref_idx = torch.tensor(np.ascontiguousarray(ref_idx))

        return scene, verts, faces, rgb, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, norm_max
