import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import draw_geometries
from pathlib import Path
import potpourri3d as pp3d
import sys
import torch

pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
# pkg_path = Path("diffusion-net", "src")
sys.path.append(pkg_path.as_posix())
import diffusion_net



data_dir = Path("/media/cychen/HDD/scannet")



# Visualize proportion of vertices after mesh simplification
def draw_nvertices_hist(out_path):

    vf = np.loadtxt(data_dir/ "stats"/ "nvnf_centered.txt", delimiter=',', skiprows=1, dtype=np.int_)
    v002 = np.loadtxt(data_dir/ "stats"/ "nvnf_level1.txt", delimiter=',', skiprows=1, dtype=np.int_)
    _, ax = plt.subplots(layout="constrained")
    _ = ax.hist(vf[:,0], bins=200, alpha=0.7, label="Raw Scenes")
    _ = ax.hist(v002, bins=200, alpha=0.7, label="Simplified Scenes")
    ax.set_title("Number of Vertices in ScanNet Scenes", fontsize=16)
    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Number of Scenes")
    ax.legend(fontsize=8)
    plt.savefig(out_path)
    plt.close()

    return

out_dir = Path("visualizations", "metrics")
# draw_nvertices_hist(out_dir/ "n_vertices_hist.svg")



# Visualize eigenbases on (dis-)connected meshes

# split_dir = Path(__file__).parent/ "splits"
split_dir = Path("gdl_scene_segment", "datasets", "splits")
with open(split_dir/ "scannetv2_train.txt", 'r') as f:
    scenes = f.read().splitlines()
with open(split_dir/ "scannetv2_val.txt", 'r') as f:
    scenes.extend(f.read().splitlines())

centered = np.loadtxt(data_dir/ "stats"/ "nvnf_centered.txt", dtype=np.int_)
single = np.loadtxt(data_dir/ "stats"/ "nvnf_single_components.txt", dtype=np.int_)

def draw_single_comp_proportion(out_path, centered, single):

    _, ax = plt.subplots(layout="constrained")
    _ = ax.hist((single/centered)[:,0], bins=100, alpha=0.9)
    ax.set_title("Largest Connected Component's Proportion", fontsize=16)
    ax.set_xlabel("Proportion of Vertices")
    ax.set_ylabel("Number of Scenes")
    plt.savefig(out_path)
    plt.close()

    return

out_dir = Path("visualizations", "eigenbases")
# draw_single_comp_proportion(out_dir/ "largest_comp_proportion.svg", centered, single)



def get_operators_from_idx(data_dir:Path,
                           preprocess:str,
                           scenes:list[str],
                           idx:int,
                           ):

    scene = scenes[idx]
    mesh_path = data_dir/ preprocess/ "scenes"/ f"{scene}_vh_clean_2.ply"
    verts, faces = pp3d.read_mesh(mesh_path.as_posix())
    verts = torch.tensor(np.ascontiguousarray(verts)).float()
    faces = torch.tensor(np.ascontiguousarray(faces.astype(np.int32)))

    with open(data_dir/ "centered"/ "norm_max"/ f"{scene}_norm_max.txt", 'r') as f:
        norm_max = float(f.read())
    verts = verts / norm_max

    op_cache_dir = data_dir/ "diffusion-net"/ "op_cache_128"/ preprocess
    _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces, 128, op_cache_dir)

    return verts, faces, mass, L, evals, evecs, gradX, gradY

def visualize_raw_mesh(data_dir:Path,
                       preprocess:str,
                       scenes:list[str],
                       idx:int,
                       ) -> None:

    scene = scenes[idx]
    mesh_path = data_dir/ preprocess/ "scenes"/ f"{scene}_vh_clean_2.ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
    with open(data_dir/ "centered"/ "norm_max"/ f"{scene}_norm_max.txt", 'r') as f:
        norm_max = float(f.read())
    mesh = mesh.scale(1/norm_max, (0,0,0))
    draw_geometries([mesh])

    return

def visualize_eigenbasis(mesh,
                         evecs:torch.Tensor,
                         ) -> None:

    viridis = mpl.colormaps['viridis']
    evecs_normal = torch.clamp((evecs+1)/2, min=0, max=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(viridis(evecs_normal)[:,:3])

    draw_geometries([mesh])

    return

def visualize_raw_eigenbasis(data_dir:Path,
                             preprocess:str,
                             scenes:list[str],
                             idx:int,
                             ) -> None:

    ops = get_operators_from_idx(data_dir, preprocess, scenes, idx)
    visualize_raw_mesh(data_dir, preprocess, scenes, idx)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(ops[0])
    mesh.triangles = o3d.utility.Vector3iVector(ops[1])
    # eig_k = [1,4,7,8,10,13,24,31]
    eig_k = range(128)
    for i in eig_k:
        print(i)
        visualize_eigenbasis(mesh, ops[5][:,i])
    
    return

# idx = np.argwhere((single/centered)[:,0] == 1).flatten()[2]
# idx = np.abs((single/centered)[:,0] - 0.9).argmin()
# idx = np.abs((single/centered)[:,0] - 0.5).argmin()
# idx = (single/centered)[:,0].argmin()
# visualize_raw_eigenbasis(data_dir, "centered", scenes, idx)
