import argparse
import numpy as np
from pathlib import Path
import potpourri3d as pp3d
import sys
import torch
from tqdm import tqdm

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(pkg_path.as_posix())
import diffusion_net



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="directory to the ScanNet dataset")
    parser.add_argument("--preprocess", type=str,
                        help="which preprocessing", required=True)
    args = parser.parse_args()

    # model settings
    k_eig = 128

    # paths
    data_dir = Path(args.data_dir, args.preprocess)
    op_cache_dir = data_dir.parent/ "diffusion-net"/ f"op_cache_{k_eig}"/ args.preprocess
    op_cache_dir.mkdir(parents=True, exist_ok=True)

    # load splits
    split_dir = Path(__file__).parent/ "splits"
    with open(split_dir/ "scannetv2_train.txt", 'r') as f:
        scenes = f.read().splitlines()
    with open(split_dir/ "scannetv2_val.txt", 'r') as f:
        scenes.extend(f.read().splitlines())
    with open(split_dir/ "scannetv2_test.txt", 'r') as f:
        scenes.extend(f.read().splitlines())

    for scene in tqdm(scenes):

        # load mesh
        mesh_path = data_dir/ "scenes"/ f"{scene}_vh_clean_2.ply"
        verts, faces = pp3d.read_mesh(mesh_path.as_posix())
        verts = torch.tensor(np.ascontiguousarray(verts)).float()
        faces = torch.tensor(np.ascontiguousarray(faces.astype(np.int32)))

        # unit scale
        scale = np.linalg.norm(verts, axis=-1).max()
        verts = verts / scale
        # verts = diffusion_net.geometry.normalize_positions(verts)

        # precompute operators
        _ = diffusion_net.geometry.get_operators(verts, faces, k_eig, op_cache_dir)
