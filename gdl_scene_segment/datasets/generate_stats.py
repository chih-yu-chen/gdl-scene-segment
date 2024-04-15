import numpy as np
from pathlib import Path
import potpourri3d as pp3d
from tqdm import tqdm



data_dir = Path("/media/cychen/HDD/scannet")
split_dir = Path(__file__).parent/ "splits"
# split_dir = Path("gdl_scene_segment", "datasets", "splits")
with open(split_dir/ "scannetv2_train.txt", 'r') as f:
    scenes = f.read().splitlines()
with open(split_dir/ "scannetv2_val.txt", 'r') as f:
    scenes.extend(f.read().splitlines())



# Mesh Pre-processing
def generate_preprocess_stats(data_dir, scenes, preprocesses):

    for preprocess in preprocesses:

        verts = []
        faces = []

        for scene in tqdm(scenes):
            mesh_path = data_dir/ preprocess/ "scenes"/ f"{scene}_vh_clean_2.ply"
            v, f = pp3d.read_mesh(mesh_path.as_posix())
            verts.append(v.shape[0])
            faces.append(f.shape[0])

        np.savetxt(data_dir/ "stats"/ f"nvnf_{preprocess}.txt", np.vstack((verts, faces)).T, fmt='%d')

preprocesses = ["centered", "single_components", "filled_multi", "holes_filled"]
generate_preprocess_stats(data_dir, scenes, preprocesses)



# Mesh Simplification
def generate_simplification_stats(data_dir, scenes, n_levels, preprocess):

    for i in range(n_levels):

        verts = []
        faces = []

        for scene in tqdm(scenes):
            mesh_path = data_dir/ preprocess/ "hierarchy"/ "scenes"/ f"{scene}_vh_clean_2_{i+1}.ply"
            v, f = pp3d.read_mesh(mesh_path.as_posix())
            verts.append(v.shape[0])
            faces.append(f.shape[0])

        np.savetxt(data_dir/ "stats"/ f"nvnf_level{i+1}.txt", np.vstack((verts, faces)).T, fmt='%d')

n_levels = 4
generate_simplification_stats(data_dir, scenes, n_levels, "centered")
