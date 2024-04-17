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



def compute_gradients(verts, faces, L, normals=None):

    device = verts.device
    dtype = verts.dtype

    # build gradient matrices
    frames = diffusion_net.geometry.build_tangent_frames(verts, faces, normals=normals)
    inds_row, inds_col = L.indices()
    edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
    edge_vecs = diffusion_net.geometry.edge_tangent_vectors(verts, frames, edges)
    grad_mat_np = diffusion_net.geometry.build_grad(verts, edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # convert back to torch
    gradX = diffusion_net.utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = diffusion_net.utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return gradX, gradY



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="directory to the ScanNet dataset")
    parser.add_argument("--preprocess", type=str,
                        help="which preprocessing", required=True)
    parser.add_argument('--n_levels', type=int, default=0,
                        help="how many levels for architecture; input 0 if vanilla DiffusionNet")
    args = parser.parse_args()

    # model settings
    k_eig = 128
    n_levels = args.n_levels

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
        mesh_paths = [data_dir/ "scenes"/ f"{scene}_vh_clean_2.ply"]

        if n_levels > 0:
            mesh_paths += [hierarchy_dir/ "scenes"/ f"{scene}_vh_clean_2_{i+1}.ply"
                           for i in range(n_levels)]

        verts = []
        faces = []

        for path in mesh_paths:
            v, f = pp3d.read_mesh(path.as_posix())
            verts.append(torch.tensor(np.ascontiguousarray(v)).float())
            faces.append(torch.tensor(np.ascontiguousarray(f.astype(np.int32))))

        # unit scale
        norm_max = np.linalg.norm(verts[0], axis=-1).max()
        with open(normmax_dir/ f"{scene}_norm_max.txt", 'w') as f:
            f.write(str(norm_max))
        verts = [v / norm_max for v in verts]

        # precompute operators
        for v, f in zip(verts, faces):
            _ = diffusion_net.geometry.get_operators(v, f, k_eig, op_cache_dir)
