import numpy as np
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

def get_ious(preds, labels, n_class):

    mcm = MulticlassConfusionMatrix(n_class)
    if preds.device != "cpu":
        preds = preds.cpu()
    if labels.device != "cpu":
        labels = labels.cpu()
    mcm.update(preds, labels)
    mcm_results = mcm.compute()
    tps = mcm_results.diagonal()
    fps = mcm_results.sum(axis=1) - tps
    fns = mcm_results.sum(axis=0) - tps

    return tps, fps, fns
