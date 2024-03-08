import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset

import potpourri3d as pp3d
from torch_geometric.typing import OptTensor

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(pkg_path.as_posix())
import diffusion_net



class ScanNetDataGeometric(Data):

    def __init__(self,
                 verts: Tensor,
                 faces: Tensor,
                 rgb: Tensor,
                 labels: Tensor,
                 ref_idx: Tensor):
        
        super().__init__(x=rgb, y=labels, pos=verts)
        self.face = faces.T
        self.reference_index = ref_idx

        return

class ScanNetDataGeometricProcessed(Data):

    def __init__(self,
                 x: Tensor | None = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 y: OptTensor = None,
                 pos: OptTensor = None,
                 **kwargs):
        
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

class ScanNetDatasetGeometric(Dataset):

    def __init__(self,
                 train: bool,
                 root: str,
                 preprocess: str,
                 k_eig: int,
                 op_cache_dir: Path,
                 transform: [Callable[..., Any], None] = None,
                 pre_filter: [Callable[..., Any], None] = None,
                 log: bool = True):
        
        super().__init__(root=root, transform=transform, pre_filter=pre_filter, log=log)
        self.train = train
        self.data_dir = root/ preprocess
        self.preprocess = preprocess
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir/ preprocess
        self.pre_transform = pre_transform
        self.classes = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        self.label_map = np.ones(41, dtype=np.int8) * -100
        np.put(self.label_map, self.classes, np.arange(self.classes.size))
        self.scene_list = self._load_split()

        return

    def len(self):
        
        return len(self.scene_list)
    
    def get(self, idx):

        labels = self.label_map[labels]
        labels = torch.tensor(np.ascontiguousarray(labels.astype(np.int64)))


        return

    def _load_split(self):
        
        split_dir = Path(__file__).parent/ "splits"
        if self.train:
            with open(split_dir/ "scannetv2_train.txt", 'r') as f:
            # with open(split_dir/ "scannetv2_train_3e5.txt", 'r') as f:
                scene_list = f.read().splitlines()
        else:
            with open(split_dir/ "scannetv2_val.txt", 'r') as f:
                scene_list = f.read().splitlines()

        return scene_list

def pre_transform(verts, faces, k_eig, op_cache_dir):

    _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, k_eig, op_cache_dir)

    return mass, L, evals, evecs, gradX, gradY
