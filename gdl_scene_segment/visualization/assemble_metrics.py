import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union



def loss_iou_to_metrics(exp_dir: Path) -> None:

    loss = np.loadtxt(exp_dir/ "loss.csv", delimiter=',')
    train = np.loadtxt(exp_dir/ "train_iou.csv", delimiter=',', skiprows=1)
    val = np.loadtxt(exp_dir/ "val_iou.csv", delimiter=',', skiprows=1)

    header = "Train Loss,Val Loss,Train mIoU,Val mIoU"
    metrics = np.hstack((loss, train[:,0:1], val[:,0:1]))
    np.savetxt(exp_dir/ "metrics.csv", metrics, delimiter=',', fmt='%.4f', header=header, comments="")

    return

def read_val_ious(exp_dir: Path) -> np.ndarray:

    val = np.loadtxt(exp_dir/ "val_iou.csv", delimiter=',', skiprows=1)
    n_col = val.shape[1]
    if n_col == 21:
        return val
    if n_col == 22:
        return np.hstack((val[:,(0,)], val[:,2:]))
    if n_col > 22:
        raise ValueError(f"The csv file has {n_col} columns, some of which are not identified")

def exps_to_csv(exps:list[np.ndarray],
                n_epochs:Union[int,list[int]],
                indices:list[str],
                out_path:Path
                ) -> None:

    if isinstance(n_epochs, int):
        n_epochs = [n_epochs] * len(exps)

    exps = get_max_mious(exps, n_epochs)
    df = pd.DataFrame(exps, index=indices, columns=class_names)
    df.to_csv(f"{out_path}.csv", float_format='%.3f', index_label="Experiment Condition")

    return

def get_max_mious(exps:list[np.ndarray],
                  n_epochs:list[int],
                  ) -> np.ndarray:

    return [exp[exp[:n,0].argmax()] for exp, n in zip(exps, n_epochs)]



# Settings
class_names = "mIoU,wall,floor,cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refridgerator,shower curtain,toilet,sink,bathtub,otherfurniture"
class_names = class_names.split(",")

# Paths
exps_dir = Path("experiments")
base_dir = exps_dir/ "baseline"
hier_dir = exps_dir/ "hierarchy"



# Experiments on Vanilla DiffusionNet
# --------------------------------------------
xyzrgb = read_val_ious(base_dir/ "00_baseline_xyzrgb")
noGradRot = read_val_ious(base_dir/ "03_baseline_noGradientRotation")

# Inputs
xyz = read_val_ious(base_dir/ "00_baseline_xyz")
hks = read_val_ious(base_dir/ "00_baseline_hks")
hksrgb = read_val_ious(base_dir/ "00_baseline_hksrgb")
exps = [xyz, xyzrgb, hks, hksrgb]
indices = ("XYZ", "XYZRGB", "HKS", "HKSRGB")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp00")

single = read_val_ious(base_dir/ "01_baseline_singleComponents")
filled = read_val_ious(base_dir/ "01_baseline_holesFilled")
multi = read_val_ious(base_dir/ "01_baseline_filledMulti")
exps = [xyzrgb, single, filled, multi]
indices = ("Raw", "Disconnection Removed", "Holes Filled", "Both")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp01")

vc002 = read_val_ious(base_dir/ "02_baseline_vc002")
exps = [xyzrgb, vc002]
indices = ("Raw", "Vertex-Clustered")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp02")

exps = [xyzrgb, noGradRot]
indices = ("Grad. Rot.", "No Grad. Rot.")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp03")

rotAug = read_val_ious(base_dir/ "04_baseline_rotAug")
exps = [xyzrgb, rotAug]
indices = ("All Aug.", "Rot. Aug.")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp04")

lr01 = read_val_ious(base_dir/ "05_baseline_lr01")
lr001 = read_val_ious(base_dir/ "05_baseline_lr001")
exps = [noGradRot, lr001, lr01]
indices = ("LR 0.001", "LR 0.01", "LR 0.1 (10)")
exps_to_csv(exps, 50, indices, exps_dir/ "base_exp05_0")

lrs_diffnet = read_val_ious(base_dir/ "06_baseline_lrs_DiffusionNet_50")
lrs_vmnet = read_val_ious(base_dir/ "06_baseline_lrs_VMNet_100")
exps = [noGradRot, lrs_diffnet, lrs_vmnet]
indices = ("PicassoNet++ (75)", "DiffusionNet (75)", "VMNet (75)")
exps_to_csv(exps, 75, indices, exps_dir/ "base_exp05_1")

cw64 = read_val_ious(base_dir/ "07_baseline_cWidth64")
cw256 = read_val_ious(base_dir/ "07_baseline_cWidth256")
block8 = read_val_ious(base_dir/ "08_baseline_8Blocks_cWidth64")
exps = [cw64, noGradRot, cw256, block8]
indices = ("4 Blocks Width 64 (75)", "4 Blocks Width 128 (75)", "4 Blocks Width 256", "8 Blocks Width 64")
exps_to_csv(exps, [75,75,50,50], indices, exps_dir/ "base_exp05_2-3")



# Experiments on Proposed Architecture
# --------------------------------------------
noGradRot = read_val_ious(base_dir/ "03_baseline_noGradientRotation")
hierarchy = read_val_ious(hier_dir/ "00_hierarchy")
hierarchy_geo = read_val_ious(hier_dir/ "01_hierarchy_geo")
hierarchy_2Levels = read_val_ious(hier_dir/ "03_hierarchy_2Levels")

exps = [noGradRot, hierarchy]
indices = ("Vanilla", "Proposed")
exps_to_csv(exps, 50, indices, exps_dir/ "hier_exp00")

hierarchy_euc = read_val_ious(hier_dir/ "01_hierarchy_euc")
exps = [hierarchy, hierarchy_euc, hierarchy_geo]
indices = ("Both", "Euclidean (30)", "Geodesic (10)")
exps_to_csv(exps, 50, indices, exps_dir/ "hier_exp01")

hierarchy_resLinear = read_val_ious(hier_dir/ "02_hierarchy_resLinear")
hierarchy_noRes = read_val_ious(hier_dir/ "02_hierarchy_noRes")
exps = [hierarchy, hierarchy_resLinear, hierarchy_noRes]
indices = ("Proposed", "Linear Residual", "No Residual")
exps_to_csv(exps, 50, indices, exps_dir/ "hier_exp02")

hierarchy_1Level = read_val_ious(hier_dir/ "03_hierarchy_1Level")
exps = [noGradRot, hierarchy_geo, hierarchy_2Levels, hierarchy_1Level]
indices = ("Vanilla", "4 Levels (10)", "2 Levels", "1 Level")
exps_to_csv(exps, 50, indices, exps_dir/ "hier_exp03")

hierarchy_2Levels_noAug = read_val_ious(hier_dir/ "04_hierarchy_2Levels_noAug")
hierarchy_2Levels_rotAug = read_val_ious(hier_dir/ "04_hierarchy_2Levels_rotAug")
hierarchy_2Levels_noSkip = read_val_ious(hier_dir/ "04_hierarchy_2Levels_noSkip")
hierarchy_2Levels_lr00001 = read_val_ious(hier_dir/ "04_hierarchy_2Levels_lr00001-4_noGradRot")
exps = [hierarchy_2Levels, hierarchy_2Levels_noAug, hierarchy_2Levels_rotAug, hierarchy_2Levels_noSkip, hierarchy_2Levels_lr00001]
indices = ("Default", "No Aug.", "Rot. Aug. (38)", "No Skip (8)", "LR 0.0001 (37)")
exps_to_csv(exps, 50, indices, exps_dir/ "hier_exp04")
