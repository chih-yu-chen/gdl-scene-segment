import numpy as np
from pathlib import Path
from plyfile import PlyData
import torch
from torcheval.metrics import MulticlassConfusionMatrix



def random_rotate_points_z(pts, prob=0.95):

    if np.random.rand() < prob:
        angles = np.random.rand() * 2 * np.pi
        rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
        rot_mats[0,0] = np.cos(angles)
        rot_mats[0,1] = np.sin(angles)
        rot_mats[1,0] = -np.sin(angles)
        rot_mats[1,1] = np.cos(angles)
        rot_mats[2,2] = 1
        pts = torch.matmul(pts, rot_mats)

    return pts

def random_translate(pts, scale=1, prob=0.95):

    if np.random.rand() < prob:
        offset = (np.random.rand(3).astype(np.float32) - 0.5) * scale
        pts += offset

    return pts

def random_flip(pts, prob=0.95):

    if np.random.rand() < prob:
        sign = np.random.randint(0,2) * 2 - 1
        pts[:,0] *= sign

    return pts

def random_scale(pts, max_scale=50, prob=0.95):

    if np.random.rand() < prob:
        scale = np.random.rand() * max_scale
        pts *= scale

    return pts

def random_rgb_jitter(rgb, scale=0.05, prob=0.95):

    if np.random.rand() < prob:
        jitter = np.random.normal(size=rgb.shape).astype(np.float32) * scale
        rgb += jitter
        rgb = torch.clamp(rgb, min=0, max=1)

    return rgb

def get_ious(preds, labels, n_class, device):

    mcm = MulticlassConfusionMatrix(n_class).to(device)
    mcm.update(preds, labels)
    mcm_results = mcm.compute()
    tps = mcm_results.diagonal()
    fps = mcm_results.sum(axis=1) - tps
    fns = mcm_results.sum(axis=0) - tps

    return tps, fps, fns

def back_convert_labels(pred_dir, classes):

    label_files = Path(pred_dir).glob("*.txt")
    for file in label_files:
        labels = np.loadtxt(file).astype(np.uint8)
        # np.unique(labels, return_counts=True)
        labels = classes[labels]
        print(f"saving back-converted labels at {file}")
        np.savetxt(file, labels, fmt='%d', delimiter='\n')

def save_ground_truth_labels(scene_list, gt_dir, dst_dir, label_map):

    for scene in scene_list:
        label_path = gt_dir/scene/(scene+"_vh_clean_2.labels.ply")
        with open(label_path, 'rb') as f:
            labels = PlyData.read(f)['vertex'].data['label']
        labels = label_map[labels]
        dst_path = dst_dir/(scene+"_labels.txt")
        print(f"saving ground truth labels at {dst_path}")
        np.savetxt(dst_path, labels, fmt='%d', delimiter='\n')

    return

def save_rgb(scene_list, mesh_dir, dst_dir):

    for scene in scene_list:
        mesh_path = mesh_dir/scene/(scene+"_vh_clean_2.ply")
        with open(mesh_path, 'rb') as f:
            plydata = PlyData.read(f)
        rgb = np.vstack((plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue'])).T
        dst_path = dst_dir/(scene+"_rgb.txt")
        print(f"saving rgb of mesh at {dst_path}")
        np.savetxt(dst_path, rgb, fmt='%d', delimiter=',')

    return


if __name__ == '__main__':

    # classes = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    # label_map = np.zeros(41, dtype=np.int8)
    # np.put(label_map, classes, np.arange(classes.size))

    val_split = "ScanNet/Tasks/Benchmark/scannetv2_test.txt"
    with open(val_split, 'r') as f:
        scene_list = f.read().splitlines()

    # pred_dir = Path("/media/cychen/HDD/scannet/gts/")
    # back_convert_labels(pred_dir, classes)

    # gt_dir = Path("/media/cychen/HDD/scannet/scans")
    # dst_dir = Path("/media/cychen/HDD/scannet/gts")
    # save_ground_truth_labels(scene_list, gt_dir, dst_dir, label_map)

    mesh_dir = Path("/media/cychen/HDD/scannet/scans_test")
    dst_dir = Path("/media/cychen/HDD/scannet/rgb")
    save_rgb(scene_list, mesh_dir, dst_dir)
