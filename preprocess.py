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

def get_referenced_idx(mesh):
    
    triangles = np.asarray(mesh.triangles)

    return np.unique(triangles.flatten())


def fill_holes(mesh, hole_size):

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = o3d.t.geometry.TriangleMesh.fill_holes(mesh, hole_size=hole_size)
    filled = filled.to_legacy()

    return filled

def get_referenced_rgb(mesh):

    rgb = np.asarray(mesh.vertex_colors) * 255

    return rgb.astype(np.uint8)

def get_referenced_labels(mesh_dir, scene, idx):

    with open(mesh_dir / scene / f"{scene}_vh_clean_2.labels.ply", 'rb') as f:
        labels = PlyData.read(f)['vertex'].data['label']

    return labels[idx]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, help="which machine", required=True)
    parser.add_argument("--dst", type=str, help="destination directory", required=True)
    parser.add_argument("--test", action="store_true", help="preprocess the test set, else preprocess the training and validation sets")
    parser.add_argument("--raw", action="store_true", help="do not preprocess, only extract rgb and labels from raw")
    parser.add_argument("--remove_disconnection", action="store_true", help="remove disconnected components, leaving only one component per mesh")
    parser.add_argument("--fill_holes", action="store", type=float, dest="hole_size", default=0, help="fill holes with shorter radius than the provided value")
    args = parser.parse_args()

    # set paths
    if args.machine == "room":
        mesh_dir = Path("/media/cychen/HDD/scannet/scans")
    elif args.machine == "hal":
        mesh_dir = Path("/shared/scannet/scans")
    if args.test:
        mesh_dir = Path(mesh_dir.as_posix() + "_test")

    dst_dir = Path(args.dst)
    (dst_dir / "idx").mkdir(parents=True, exist_ok=True)
    (dst_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)

    # read scene list
    if args.test:
        split_test = "ScanNet/Tasks/Benchmark/scannetv2_test.txt"
        with open(split_test, 'r') as f:
            scenes = f.read().splitlines()
    else:
        split_train = "ScanNet/Tasks/Benchmark/scannetv2_train.txt"
        split_val = "ScanNet/Tasks/Benchmark/scannetv2_val.txt"
        with open(split_train, 'r') as f:
            scenes = f.read().splitlines()
        with open(split_val, 'r') as f:
            scenes.extend(f.read().splitlines())

    # preprocess scenes
    for scene in tqdm(scenes):

        mesh_path = mesh_dir / scene / f"{scene}_vh_clean_2.ply"
        mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())

        if args.raw:
            ref_idx = np.arange(np.asarray(mesh.vertices).shape[0])

        else:
            mesh = mesh.remove_non_manifold_edges()

            if args.remove_disconnection:
                mesh = remove_disconnected_components(mesh)

            if args.hole_size > 0:
                mesh = fill_holes(mesh, args.hole_size)

            ref_idx = get_referenced_idx(mesh)
            mesh = mesh.remove_unreferenced_vertices()

        rgb = get_referenced_rgb(mesh)

        if not args.test:
            labels = get_referenced_labels(mesh_dir, scene, ref_idx)

        if not args.raw:
            o3d.io.write_triangle_mesh((dst_dir / f"{scene}_vh_clean_2.ply").as_posix(), mesh)
            np.savetxt(dst_dir / "idx" / f"{scene}_referenced_idx.txt", ref_idx, fmt='%d', delimiter=',')

        np.savetxt(dst_dir / "rgb" / f"{scene}_rgb.txt", rgb, fmt='%d', delimiter=',')

        if not args.test:
            np.savetxt(dst_dir / "labels" / f"{scene}_labels.txt", labels, fmt='%d', delimiter=',')
