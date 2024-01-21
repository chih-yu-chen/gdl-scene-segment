import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from plyfile import PlyData
from tqdm import tqdm



def remove_disconnected_components(mesh):

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < cluster_n_triangles.max()
    mesh.remove_triangles_by_mask(triangles_to_remove)

    return mesh

def fill_holes(mesh, hole_size):

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = o3d.t.geometry.TriangleMesh.fill_holes(mesh, hole_size=hole_size)
    filled = filled.to_legacy()

    return filled

def get_referenced_idx(mesh):
    
    triangles = np.asarray(mesh.triangles)

    return np.unique(triangles.flatten())

def get_referenced_rgb(mesh):

    rgb = np.asarray(mesh.vertex_colors) * 255

    return rgb.astype(np.uint8)

def get_referenced_labels(mesh_path: Path, idx):

    with open(mesh_path.with_suffix(".labels.ply"), 'rb') as f:
        labels = PlyData.read(f)['vertex'].data['label']

    return labels[idx]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="directory to the ScanNet dataset")
    parser.add_argument("--dst", type=str, help="relative destination directory", required=True)
    parser.add_argument("--test", action="store_true",
                        help="preprocess the test set, else preprocess the training and validation sets")
    parser.add_argument("--remove_disconnection", action="store_true",
                        help="remove disconnected components, leaving only one component per mesh")
    parser.add_argument("--fill_holes", action="store", type=float, dest="hole_size", default=0,
                        help="fill holes with shorter radius than the provided value")
    args = parser.parse_args()

    # set paths
    data_dir = Path(args.data_dir)
    dst_dir = data_dir/ args.dst
    (dst_dir/ "scenes").mkdir(parents=True, exist_ok=True)
    (dst_dir/ "idx").mkdir(parents=True, exist_ok=True)
    (dst_dir/ "rgb").mkdir(parents=True, exist_ok=True)
    (dst_dir/ "labels").mkdir(parents=True, exist_ok=True)

    # read scene list
    split_dir = Path(__file__).parent/ "splits"
    if args.test:
        with open(split_dir/ "scannetv2_test.txt", 'r') as f:
            scenes = f.read().splitlines()
    else:
        with open(split_dir/ "scannetv2_train.txt", 'r') as f:
            scenes = f.read().splitlines()
        with open(split_dir/ "scannetv2_val.txt", 'r') as f:
            scenes.extend(f.read().splitlines())

    remove_disconnection = True if args.hole_size > 0 else args.remove_disconnection

    # preprocess scenes
    for scene in tqdm(scenes):

        if args.test:
            mesh_path = data_dir/ "scans_test"/ scene/ f"{scene}_vh_clean_2.ply"
        else:
            mesh_path = data_dir/ "scans"/ scene/ f"{scene}_vh_clean_2.ply"
        mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
        ref_idx = np.arange(np.asarray(mesh.vertices).shape[0])

        means = mesh.get_center()
        mesh = mesh.translate(-means)

        if remove_disconnection:

            mesh = mesh.remove_non_manifold_edges()
            mesh = remove_disconnected_components(mesh)

            if args.hole_size > 0:
                mesh = fill_holes(mesh, args.hole_size)

            ref_idx = get_referenced_idx(mesh)
            np.savetxt(dst_dir/ "idx"/ f"{scene}_referenced_idx.txt", ref_idx, fmt='%d', delimiter=',')
            mesh = mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh((dst_dir/ "scenes"/ f"{scene}_vh_clean_2.ply").as_posix(), mesh)

        rgb = get_referenced_rgb(mesh)
        np.savetxt(dst_dir/ "rgb"/ f"{scene}_rgb.txt", rgb, fmt='%d', delimiter=',')

        if not args.test:
            labels = get_referenced_labels(mesh_path, ref_idx)
            np.savetxt(dst_dir/ "labels"/ f"{scene}_labels.txt", labels, fmt='%d', delimiter=',')
