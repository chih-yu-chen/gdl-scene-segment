import argparse
from functools import partial
import multiprocessing as mp
import numpy as np
from pathlib import Path
import potpourri3d as pp3d
import sys
import torch

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net



def process_frame(scene:str,
                  data_dir:Path,
                  hierarchy_dir:Path,
                  op_cache_dir:Path,
                  n_levels:int,
                  k_eig:int):

    print(f"Processing: {scene}")

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
    verts = [v / scale for v in verts]

    # precompute operators
    for v, f in zip(verts, faces):
        _ = diffusion_net.geometry.get_operators(v, f, k_eig, op_cache_dir)

    print(f"Processed: {scene}")

    return


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
    op_cache_dir = data_dir.parent/ "diffusion-net"/ f"op_cache_{k_eig}"/ args.preprocess
    op_cache_dir.mkdir(parents=True, exist_ok=True)

    # load splits
    split_dir = Path("splits")
    with open(split_dir/ "scannetv2_train.txt", 'r') as f:
        scenes = f.read().splitlines()
    with open(split_dir/ "scannetv2_val.txt", 'r') as f:
        scenes.extend(f.read().splitlines())
    with open(split_dir/ "scannetv2_test.txt", 'r') as f:
        scenes.extend(f.read().splitlines())

    # Partial function
    process_frame_p = partial(process_frame,
                              data_dir=data_dir,
                              hierarchy_dir=hierarchy_dir,
                              op_cache_dir=op_cache_dir,
                              n_levels=n_levels,
                              k_eig=k_eig)

    # multi-processing
    pf_pool = mp.Pool(processes=16)
    pf_pool.map(process_frame_p, scenes)
    pf_pool.close()
    pf_pool.join()
