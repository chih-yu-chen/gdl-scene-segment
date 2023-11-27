import argparse
from collections import defaultdict
import csv
from functools import partial
import multiprocessing as mp
import numpy as np
import open3d as o3d
import os
from pathlib import Path
from sklearn.neighbors import BallTree



def read_qem_csv(csv_path: Path
                 ) -> tuple[np.ndarray, list[np.ndarray]]:
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        coords = list(reader)
    
    new_coords = np.asarray([row[:3] for row in coords], dtype=np.float_).reshape((-1,3))
    old_coords_list = [np.asarray(row[3:], dtype=np.float_).reshape((-1,3)) for row in coords]

    return new_coords, old_coords_list

def get_nearest_neighbor_ids(verts: np.ndarray,
                             coords_list: list[np.ndarray]
                             ) -> list[np.ndarray]:

    balltree = BallTree(verts)
    ids_list = [balltree.query(coords, k=1, return_distance=False).flatten()
                for coords in coords_list]

    return ids_list

def update_tracemap(trace_map: dict[int, list],
                    new_ids: np.ndarray,
                    old_ids: list[np.ndarray]
                    ) -> dict[int, list]:

    for new_id, old_id in zip(new_ids, old_ids):
        if isinstance(old_id, np.ndarray):
            trace_map[new_id].extend(old_id.tolist())
        else:
            trace_map[new_id].append(old_id)
    
    return trace_map

def get_qem_tracemap(csv_path: Path,
                     new_verts: np.ndarray,
                     old_verts: np.ndarray
                     ) -> np.ndarray:

    map_new2old = defaultdict(list)

    new_coords, old_coords_list = read_qem_csv(csv_path)
    new_ids = get_nearest_neighbor_ids(new_verts, [new_coords])[0]
    old_ids_list = get_nearest_neighbor_ids(old_verts, old_coords_list)

    old_ids = [i for ids in old_ids_list for i in ids]
    if new_ids.size != np.unique(new_ids).size:
        raise ValueError("GRAPH LEVEL GENERATION ERROR: duplicate new_ids")
    if len(old_ids) != np.unique(old_ids).size:
        raise ValueError("GRAPH LEVEL GENERATION ERROR: duplicate old_ids")

    map_new2old = update_tracemap(map_new2old, new_ids, old_ids_list)

    old_ids_left = list(set(range(old_verts.shape[0])) - set(old_ids))
    if old_ids_left:
        new_ids = get_nearest_neighbor_ids(new_verts, [old_verts[old_ids_left]])[0]
        map_new2old = update_tracemap(map_new2old, new_ids, old_ids_left)

    old_ids = [i for old in map_new2old.values() for i in old]
    if len(old_ids) != np.unique(old_ids).size:
        raise ValueError("GRAPH LEVEL GENERATION ERROR: duplicate old_ids")
    assert new_verts.shape[0] == len(map_new2old.keys()), "missing new_ids"
    assert old_verts.shape[0] == len(old_ids), "missing old_ids"

    map_old2new = np.ones(old_verts.shape[0], dtype=np.int32) * -1
    for new_id, old_ids in map_new2old.items():
        np.put(map_old2new, old_ids, new_id)

    assert not np.argwhere(map_old2new == -1).any(), "missing old_ids"

    return map_old2new

def vertex_clustering(in_path: Path,
                      out_path: Path,
                      voxel_size: float,
                      old_verts: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:

    os.system(f"../vcglib/apps/sample/trimesh_clustering/trimesh_clustering \
              {in_path} {out_path} -s {voxel_size} > /dev/null")
    
    verts = np.asarray(o3d.io.read_triangle_mesh(out_path.as_posix()).vertices)
    if not verts.any():
        raise ValueError("no vertices left")

    balltree = BallTree(verts)
    traces = balltree.query(old_verts, k=1, return_distance=False).flatten()

    return verts, traces

def quadric_error_metric(in_path:Path,
                         out_path: Path,
                         ratio: int,
                         old_verts: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:

    os.system(f"../vcglib/apps/tridecimator/tridecimator \
              {in_path} {out_path} {ratio} -On -C > /dev/null")

    verts = np.asarray(o3d.io.read_triangle_mesh(out_path.as_posix()).vertices)
    if not verts.any():
        raise ValueError("no vertices left")

    traces = get_qem_tracemap(out_path.with_suffix('.csv'), verts, old_verts)

    return verts, traces

def process_frame(scene: str,
                  scene_dir: Path,
                  preprocess: str,
                  test: bool,
                  level_params: list[float]
                  ) -> None:
    
    print(f"Processing: {scene}")

    # tmp container
    traces_list = []

    # load mesh
    mesh_path = scene_dir/ preprocess/ "scenes"/ f"{scene}_vh_clean_2.ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
    verts = np.asarray(mesh.vertices)
    
    # load label
    if not test:
        label_path = scene_dir/ preprocess/ "labels"/ f"{scene}_labels.txt"
        labels = np.loadtxt(label_path, delimiter=',', dtype=np.int8)
        balltree_l0 = BallTree(verts)

    # create output directory
    out_dir = scene_dir/ preprocess/ "hierarchy"
    (out_dir/"scenes").mkdir(parents=True, exist_ok=True)
    (out_dir/"labels").mkdir(parents=True, exist_ok=True)
    (out_dir/"traces").mkdir(parents=True, exist_ok=True)

    mesh_paths = [out_dir/"scenes"/f"{mesh_path.stem}_{i+1}.ply" for i, _ in enumerate(level_params)]
    mesh_paths.insert(0, mesh_path)

    # mesh simplification
    for i, param in enumerate(level_params):

        if param < 1:
            verts, traces = vertex_clustering(mesh_paths[i], mesh_paths[i+1],
                                       param, old_verts=verts)
        else:
            verts, traces = quadric_error_metric(mesh_paths[i], mesh_paths[i+1],
                                          int(param), old_verts=verts)
        traces_list.append(traces)

        if i == 0:
            idx = balltree_l0.query(verts, k=1, return_distance=False).flatten()
            labels_l1 = labels[idx]

    for i, traces in enumerate(traces_list):
        np.savetxt(out_dir/"traces"/f"{scene}_traces{i}{i+1}.txt", traces[:,np.newaxis], fmt='%u')
    np.savetxt(out_dir/"labels"/f"{scene}_labels1.txt", labels_l1[:,np.newaxis], fmt='%u')

    print(f"Processed: {scene}")

    return



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simplify meshes and build a hierarchy of mesh complexity')
    parser.add_argument('--split_path', type=str, required=True,
                        help="path to the training/validation/test split")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--scene_dir', type=str, required=True,
                        help="path to the directory of the scenes")
    parser.add_argument('--preprocess', type=str, required=True,
                        help="which preprocessing preceeds the simplification")
    parser.add_argument('--level_params', nargs='+', type=float, required=True,
                        help="the parameters for each level of simplification")
    args = parser.parse_args()
    
    # level_params = [0.02, 0.04, 30, 30, 30, 30, 30]

    with open(args.split_path, 'r') as f:
        scenes = f.read().splitlines()

    # Partial function
    process_frame_p = partial(process_frame,
                              scene_dir=Path(args.scene_dir), preprocess=args.preprocess, test=args.test,
                              level_params=args.level_params)

    # multi-processing
    pf_pool = mp.Pool(processes=12)
    pf_pool.map(process_frame_p, scenes)
    pf_pool.close()
    pf_pool.join()

    csv_dir = Path(args.scene_dir, args.preprocess, "hierarchy", "scenes")
    for csv_file in csv_dir.glob("*.csv"):
        csv_file.unlink()

# TODO center original mesh in preprocess.py,
# TODO don't transform original mesh in simplify_meshes.py,
# TODO scale according to original mesh or mesh_1 in scannet_dataset.py,
# TODO recompute operators for both single_components & holes_filled

# split_path = "/home/cychen/Documents/gdl-scene-segment/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
# test = False
# scene_dir = Path("/media/cychen/HDD/scannet")
# preprocess = "centered"
# level_params = [0.02, 30, 30, 30]
# i = 0
# scene = scenes[0]
# mesh
# mesh1 = o3d.io.read_triangle_mesh(mesh_paths[1].as_posix())
# mesh1
# verts1 = np.asarray(mesh1.vertices)
# traces01 = np.loadtxt(out_dir/"traces"/f"{scene}_traces{i}{i+1}.txt", dtype=np.uint32)
# uni, counts = np.unique(traces01, return_counts=True)
# uni
# counts.max()
# balltree = BallTree(verts1)
# traces = balltree.query(verts, k=1, return_distance=False).flatten()
# (traces == traces01).sum()
# traces.shape