import numpy as np
import torch
from torcheval.metrics import MulticlassConfusionMatrix



def random_rotate_points_z(prob=0.95):

    if np.random.rand() < prob:
        angles = np.random.rand() * 2 * np.pi
        rot_mat = torch.zeros(3, 3, dtype=torch.float)
        rot_mat[0,0] = np.cos(angles)
        rot_mat[0,1] = np.sin(angles)
        rot_mat[1,0] = -np.sin(angles)
        rot_mat[1,1] = np.cos(angles)
        rot_mat[2,2] = 1
    else:
        rot_mat = torch.eye(3, dtype=torch.float)

    return rot_mat

def random_translate(scale=1, prob=0.95):

    if np.random.rand() < prob:
        offset = (np.random.rand(3).astype(np.float32) - 0.5) * 2 * scale
    else:
        offset = np.zeros(3, dtype=np.float32)

    return offset

def random_flip(prob=0.95):

    if np.random.rand() < prob:
        sign = np.random.randint(0,2) * 2 - 1
    else:
        sign = 1

    return sign

def random_scale(max_scale=50, prob=0.95):

    if np.random.rand() < prob:
        scale = np.random.rand() * max_scale
    else:
        scale = 1

    return scale

def random_rgb_jitter(shape, scale=0.05, prob=0.95):

    if np.random.rand() < prob:
        jitter = np.random.normal(size=shape).astype(np.float32) * scale
    else:
        jitter = np.zeros(shape)
        rgb += jitter
        rgb = torch.clamp(rgb, min=0, max=1)

    return jitter

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
