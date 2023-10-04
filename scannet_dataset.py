import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm

import potpourri3d as pp3d
from plyfile import PlyData

pkg_path = Path(__file__).parent/"diffusion-net"/"src"
sys.path.append(str(pkg_path))
import diffusion_net



class ScanNetDataset(Dataset):

    def __init__(self, train, repo_dir, data_dir, with_rgb=False, k_eig=128, cache_dir=None, op_cache_dir=None):

        self.train = train
        self.repo_dir = Path(repo_dir)
        self.data_dir = Path(data_dir)
        self.with_rgb = with_rgb
        self.k_eig = k_eig 
        self.cache_dir = Path(cache_dir)
        self.op_cache_dir = Path(op_cache_dir)
        self.classes = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.label_map = np.ones(41, dtype=np.int8) * -100
        np.put(self.label_map, self.classes, np.arange(self.classes.size))

        # check the cache
        train_cache = self.cache_dir/"train.pt"
        test_cache = self.cache_dir/"test.pt"
        load_cache = train_cache if self.train else test_cache
        if load_cache.is_file():
            self.verts_list, self.rgb_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.scene_list = torch.load(str(load_cache))
            return

        # load train test split
        split_dir = self.repo_dir/"Tasks"/"Benchmark/"
        if self.train:
            with open(split_dir/"scannetv2_train.txt", 'r') as f:
                self.scene_list = f.read().splitlines()
        else:
            with open(split_dir/"scannetv2_val.txt", 'r') as f:
                self.scene_list = f.read().splitlines()

        # store in memory
        self.verts_list = []
        self.rgb_list = []
        self.faces_list = []
        self.frames_list =[]
        self.massvec_list = []
        self.L_list = []
        self.evals_list = []
        self.evecs_list = []
        self.gradX_list = []
        self.gradY_list = []
        self.labels_list = []

        for scene in tqdm(self.scene_list):

            # get mesh and label paths
            train_dir = self.data_dir/"scans"
            mesh_path = train_dir/scene/(scene+"_vh_clean_2.ply")
            label_path = train_dir/scene/(scene+"_vh_clean_2.labels.ply")

            # load mesh
            verts, faces = pp3d.read_mesh(str(mesh_path))
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            # load rgb
            if self.with_rgb:
                rgb_path = self.data_dir/"rgb"/f"{scene}_rgb.txt"
                rgb = np.loadtxt(rgb_path, delimiter=',')
                rgb /= 255
                rgb = torch.tensor(np.ascontiguousarray(rgb)).float()
                self.rgb_list.append(rgb)

            # precompute operators
            frames, massvec, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, self.k_eig, self.op_cache_dir)

            # load labels
            with open(label_path, 'rb') as f:
                plydata = PlyData.read(f)
            labels = plydata['vertex'].data['label']
            labels = self.label_map[labels]
            labels = torch.tensor(np.ascontiguousarray(labels.astype(np.int64)))

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.frames_list.append(frames)
            self.massvec_list.append(massvec)
            self.L_list.append(L)
            self.evals_list.append(evals)
            self.evecs_list.append(evecs)
            self.gradX_list.append(gradX)
            self.gradY_list.append(gradY)
            self.labels_list.append(labels)

        # save to cache
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save((self.verts_list, self.rgb_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list, self.labels_list, self.scene_list), str(load_cache))

        return

    def __len__(self):

        return len(self.scene_list)

    def __getitem__(self, idx):

        return self.verts_list[idx], self.rgb_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.scene_list[idx]
