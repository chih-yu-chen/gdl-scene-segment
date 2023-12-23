import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_scatter import scatter_mean

import potpourri3d as pp3d

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net



class ScanNetHierarchyDataset(Dataset):

    def __init__(self,
                 train:bool,
                 data_dir:Path,
                 preprocess:str,
                 n_levels:int,
                 with_rgb:bool,
                 k_eig:int,
                 op_cache_dir:Path):

        self.train = train
        self.data_dir = data_dir/ preprocess
        self.preprocess = preprocess
        self.hierarchy_dir = self.data_dir/ "hierarchy"
        self.with_rgb = with_rgb
        self.n_levels = n_levels
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir/ preprocess
        self.classes = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.label_map = np.ones(41, dtype=np.int8) * -100
        np.put(self.label_map, self.classes, np.arange(self.classes.size))

        # load train test split
        split_dir = Path("datasets", "splits")
        if self.train:
            with open(split_dir/ "scannetv2_train.txt", 'r') as f:
            # with open(split_dir/ "scannetv2_train_3e5.txt", 'r') as f:
                self.scene_list = f.read().splitlines()
        else:
            with open(split_dir/ "scannetv2_val.txt", 'r') as f:
                self.scene_list = f.read().splitlines()

        return

    def __len__(self):

        return len(self.scene_list)

    def __getitem__(self, idx):

        scene = self.scene_list[idx]

        # load mesh * 5
        mesh_path = self.data_dir/ "scenes"/ f"{scene}_vh_clean_2.ply"
        mesh_paths = [self.hierarchy_dir/ "scenes"/ mesh_path.with_stem(f"{mesh_path.stem}_{i+1}").name
                      for i in range(self.n_levels)]
        mesh_paths.insert(0, mesh_path)

        verts = []
        faces = []

        for path in mesh_paths:
            v, f = pp3d.read_mesh(path.as_posix())
            verts.append(torch.tensor(np.ascontiguousarray(v)).float())
            faces.append(torch.tensor(np.ascontiguousarray(f.astype(np.int32))))

        # unit scale
        scale = np.linalg.norm(verts[0], axis=-1).max()
        verts = [v / scale for v in verts]

        # load operators * 5
        mass = []
        L = []
        evals = []
        evecs = []
        gradX = []
        gradY = []
        for v, f in zip(verts[1:], faces[1:]):
            ops = diffusion_net.geometry.get_operators(v, f, self.k_eig, self.op_cache_dir)
            mass.append(ops[1])
            L.append(ops[2])
            evals.append(ops[3])
            evecs.append(ops[4])
            gradX.append(ops[5])
            gradY.append(ops[6])

        # load labels * 2
        labels = []
        label_paths = [self.data_dir/ "labels"/ f"{scene}_labels.txt",
                       self.hierarchy_dir/ "labels"/ f"{scene}_labels1.txt"]
        for path in label_paths:
            l = np.loadtxt(path, dtype=np.int8)
            l = self.label_map[l]
            labels.append(torch.tensor(np.ascontiguousarray(l.astype(np.int64))))

        # load traces * 4
        trace_paths = [self.hierarchy_dir/ "traces"/ f"{scene}_traces{i}{i+1}.txt"
                       for i in range(self.n_levels)]
        traces = [np.loadtxt(path, dtype=np.int64) for path in trace_paths]
        traces = [torch.tensor(np.ascontiguousarray(t.astype(np.int64))) for t in traces]

        # load rgb * 2
        rgbs = []
        if self.with_rgb:
            rgb_path = self.data_dir/ "rgb"/ f"{scene}_rgb.txt"
            rgb = np.loadtxt(rgb_path, delimiter=',', dtype=np.uint8)
            rgb = rgb / 255.
            rgb = torch.tensor(np.ascontiguousarray(rgb)).float()
            rgbs.append(rgb)
        rgbs.append(scatter_mean(rgb, traces[0], dim=-2))

        # load idx * 1
        if not self.preprocess == "centered":
            idx_path = self.data_dir/ "idx"/ f"{scene}_referenced_idx.txt"
            ref_idx = np.loadtxt(idx_path, dtype=np.int64)
        else:
            ref_idx = np.arange(verts[0].shape[0], dtype=np.int64)
        ref_idx = torch.tensor(np.ascontiguousarray(ref_idx))

        return scene, verts[:2], rgbs, mass, L, evals, evecs, gradX, gradY, labels, ref_idx, traces
