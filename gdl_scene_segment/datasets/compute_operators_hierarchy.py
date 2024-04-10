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
    parser.add_argument('--level_params', nargs='+', type=float, required=True,
                        help="the parameters for each level of simplification")
    args = parser.parse_args()

    # model settings
    k_eig = 128
    n_levels = len(args.level_params)

    # paths
    data_dir = Path(args.data_dir, args.preprocess)
    hierarchy_dir = data_dir/ "hierarchy"
    normmax_dir = data_dir/ "norm_max"
    normmax_dir.mkdir(parents=True, exist_ok=True)
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
        mesh_paths = [hierarchy_dir/ "scenes"/ mesh_path.with_stem(f"{mesh_path.stem}_{i+1}").name
                    for i in range(n_levels)]
        mesh_paths.insert(0, mesh_path)

        verts = []
        faces = []

        for path in mesh_paths:
            v, f = pp3d.read_mesh(path.as_posix())
            verts.append(torch.tensor(np.ascontiguousarray(v)).float())
            faces.append(torch.tensor(np.ascontiguousarray(f.astype(np.int32))))

        # unit scale
        scale = np.linalg.norm(verts[0], axis=-1).max()
        with open(normmax_dir/ f"{scene}_norm_max.txt", 'w') as f:
            f.write(str(scale))
        verts = [v / scale for v in verts]
        # verts = [diffusion_net.geometry.normalize_positions(v) for v in verts]

        # precompute operators
        for v, f in zip(verts, faces):
            _ = diffusion_net.geometry.get_operators(v, f, k_eig, op_cache_dir)
