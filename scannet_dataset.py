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

    def __init__(self, train, repo_dir, data_dir, k_eig=128, use_cache=True, op_cache_dir=None):

        self.train = train
        self.repo_dir = Path(repo_dir)
        self.data_dir = Path(data_dir)
        self.k_eig = k_eig 
        self.use_cache = use_cache
        self.cache_dir = self.data_dir/"cache"
        self.op_cache_dir = Path(op_cache_dir)
        self.n_class = 8

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex

        # check & load cache
        if self.use_cache:
            train_cache = self.cache_dir/"train.pt"
            test_cache = self.cache_dir/"test.pt"
            load_cache = train_cache if self.train else test_cache
            print(f"using dataset cache path: {load_cache}")
            if load_cache.is_file():
                print("  --> loading dataset from cache")
                self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # load train test split
        split_dir = self.repo_dir/"Tasks"/"Benchmark"
        if self.train:
            with open(split_dir/"scannetv2_train.txt", 'r') as f:
                scenes = f.read().splitlines()
            with open(split_dir/"scannetv2_val.txt", 'r') as f:
                scenes.extend(f.read().splitlines())
        else:
            with open(split_dir/"scannetv2_test.txt", 'r') as f:
                scenes = f.read().splitlines()

        # get all file paths
        mesh_paths = []
        label_paths = []

        if self.train:    
            train_dir = self.data_dir/"scans"
            for scene in scenes:
                mesh_path = train_dir/scene/(scene+"_vh_clean_2.ply")
                label_path = train_dir/scene/(scene+"_vh_clean_2.labels.ply")
                mesh_paths.append(mesh_path)
                label_paths.append(label_path)
        else:
            test_dir = self.data_dir/"scans_test"
            for scene in scenes:
                mesh_path = test_dir/scene/(scene+"_vh_clean_2.ply")
                mesh_paths.append(mesh_path)

        print(f"loading {len(mesh_paths)} meshes")

        # load
        for i, path in enumerate(mesh_paths):

            print(f"loading mesh {path}")
            verts, faces = pp3d.read_mesh(str(path))
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)

            if self.train:
                with open(label_paths[i], 'rb') as f:
                    plydata = PlyData.read(f)
                    labels = plydata['vertex'].data['label']
                labels = torch.tensor(np.ascontiguousarray(labels.astype(np.int16)))
                self.labels_list.append(labels)

        # for i, labels in enumerate(self.labels_list):
        #     self.labels_list[i] = labels

        # precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(
            self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save((self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list), load_cache)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx]
