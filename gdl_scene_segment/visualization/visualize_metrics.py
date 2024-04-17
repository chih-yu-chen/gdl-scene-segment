import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path



def read_metrics(exp_dir:Path) -> np.ndarray:

    return np.loadtxt(exp_dir/ "metrics.csv", delimiter=',', skiprows=1)

def draw_metrics(metrics:np.ndarray,
                 exp_name:list[str],
                 n_epochs:int=0,
                 ymax:float=0.,
                 out_path:Path=None,
                 ) -> None:

    len_x = [n_epochs if (n_epochs > 0) and (m.shape[0] > n_epochs) else m.shape[0] for m in metrics]
    xs = [np.arange(x) for x in len_x]

    _, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

    for i, m in enumerate(metrics):
        ax0.plot(xs[i], m[:len_x[i],0], label = f"{exp_name[i]} Train", color=tableau_colors(2*i+1))
        ax0.plot(xs[i], m[:len_x[i],1], label = f"{exp_name[i]} Val", color=tableau_colors(2*i))
        ax1.plot(xs[i], m[:len_x[i],2], label = f"{exp_name[i]} Train", color=tableau_colors(2*i+1))
        ax1.plot(xs[i], m[:len_x[i],3], label = f"{exp_name[i]} Val", color=tableau_colors(2*i))

    # ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Loss")
    # ax0.legend()
    ax0.grid(axis='both')
    if ymax > 0:
        ax0.set_ylim(0., ymax)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean Intersection over Union")
    ax1.legend(fontsize=5)
    ax1.grid(axis='both')

    plt.savefig(out_path)
    plt.close()

    return



# Plot settings
tableau_colors = mpl.colormaps['tab20']
mpl.rcParams["font.size"] = 10
mpl.rcParams['lines.linewidth'] = 0.8

# Paths
exps_dir = Path("experiments")
base_dir = exps_dir/ "baseline"
hier_dir = exps_dir/ "hierarchy"
out_dir = Path("visualizations", "metrics")
out_dir.mkdir(parents=True, exist_ok=True)



# Experiments on Vanilla DiffusionNet
# --------------------------------------------
xyzrgb = read_metrics(base_dir/ "00_baseline_xyzrgb")
noGradRot = read_metrics(base_dir/ "03_baseline_noGradientRotation")

# Inputs
xyz = read_metrics(base_dir/ "00_baseline_xyz")
hks = read_metrics(base_dir/ "00_baseline_hks")
hksrgb = read_metrics(base_dir/ "00_baseline_hksrgb")
exp_name = ["XYZ", "XYZRGB", "HKS", "HKSRGB"]
metrics = [xyz, xyzrgb, hks, hksrgb]
out_path = out_dir/ "base_exp1_model_inputs.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Preprocess
single = read_metrics(base_dir/ "01_baseline_singleComponents")
filled = read_metrics(base_dir/ "01_baseline_holesFilled")
multi = read_metrics(base_dir/ "01_baseline_filledMulti")
exp_name = ["Raw", "Disconnection Removed", "Holes Filled", "Both"]
metrics = [xyzrgb, single, filled, multi]
out_path = out_dir/ "base_exp2_mesh_preprocessing.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Exp 3-5
vc002 = read_metrics(base_dir/ "02_baseline_vc002")
rotAug = read_metrics(base_dir/ "04_baseline_rotAug")
exp_name = ["Raw/ Gradient Rotation/ All Aug.", "Vertex-Clustered", "No Gradient Rotation", "Rotational Augmentation"]
metrics = [xyzrgb, vc002, noGradRot, rotAug]
out_path = out_dir/ "base_exp3-5.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Initial Learning Rates
lr01 = read_metrics(base_dir/ "05_baseline_lr01")
lr001 = read_metrics(base_dir/ "05_baseline_lr001")
exp_name = ["LR 0.001", "LR 0.01", "LR 0.1"]
metrics = [noGradRot, lr001, lr01]
out_path = out_dir/ "base_exp6_1_ilr.svg"
draw_metrics(metrics, exp_name, 50, 2, out_path)

# Learning Rate Schedules
lrs_diffnet = read_metrics(base_dir/ "06_baseline_lrs_DiffusionNet_50")
lrs_vmnet = read_metrics(base_dir/ "06_baseline_lrs_VMNet_100")
exp_name = ["PicassoNet++", "DiffusionNet", "VMNet"]
metrics = [noGradRot, lrs_diffnet, lrs_vmnet]
out_path = out_dir/ "base_exp6_2_lrs.svg"
draw_metrics(metrics, exp_name, 75, 0, out_path)

# MLP widths
cw64 = read_metrics(base_dir/ "07_baseline_cWidth64")
cw256 = read_metrics(base_dir/ "07_baseline_cWidth256")
block8 = read_metrics(base_dir/ "08_baseline_8Blocks_cWidth64")
block16 = read_metrics(base_dir/ "08_baseline_16Blocks_cWidth64")
exp_name = ["4 Blocks, Width 64", "4 Blocks, Width 128", "4 Blocks, Width 256", "8 Blocks, Width 64", "16 Blocks, Width 64"]
metrics = [cw64, noGradRot, cw256, block8, block16]
out_path = out_dir/ "base_exp6_3-4_cWidth_blocks.svg"
draw_metrics(metrics, exp_name, 75, 0, out_path)



# Experiments on Proposed Architecture
# --------------------------------------------
hierarchy = read_metrics(hier_dir/ "00_hierarchy")
hierarchy_geo = read_metrics(hier_dir/ "01_hierarchy_geo")
hierarchy_2Levels = read_metrics(hier_dir/ "03_hierarchy_2Levels")

# First
exp_name = ["Vanilla DiffusionNet", "Proposed Architecture"]
metrics = [noGradRot, hierarchy]
out_path = out_dir/ "pyramid_exp1_whole_architecture.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Individual Branches
hierarchy_euc = read_metrics(hier_dir/ "01_hierarchy_euc")
exp_name = ["Both Branches", "Euclidean Branch", "Geodesic Branch"]
metrics = [hierarchy, hierarchy_euc, hierarchy_geo]
out_path = out_dir/ "pyramid_exp2_branches.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Structural Tweaks
hierarchy_resLinear = read_metrics(hier_dir/ "02_hierarchy_resLinear")
hierarchy_noRes = read_metrics(hier_dir/ "02_hierarchy_noRes")
exp_name = ["Proposed Architecture", "Linear before Residual", "No Residual"]
metrics = [hierarchy, hierarchy_resLinear, hierarchy_noRes]
out_path = out_dir/ "pyramid_exp3_geo_structure.svg"
draw_metrics(metrics, exp_name, 50, 6.5, out_path)

# Number of Levels
hierarchy_1Level = read_metrics(hier_dir/ "03_hierarchy_1Level")
exp_name = ["Vanilla DiffusionNet", "4 Levels", "2 Levels", "1 Level"]
metrics = [noGradRot, hierarchy_geo, hierarchy_2Levels, hierarchy_1Level]
out_path = out_dir/ "pyramid_exp4_levels.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)

# Tweaks at 2 Levels
hierarchy_2Levels_noAug = read_metrics(hier_dir/ "04_hierarchy_2Levels_noAug")
hierarchy_2Levels_noSkip = read_metrics(hier_dir/ "04_hierarchy_2Levels_noSkip")
hierarchy_2Levels_rotAug = read_metrics(hier_dir/ "04_hierarchy_2Levels_rotAug")
hierarchy_2Levels_lr00001 = read_metrics(hier_dir/ "04_hierarchy_2Levels_lr00001")
exp_name = ["Default", "No Augmentation", "Rotational Augmentation", "No Skip Connection", "Initial LR 0.0001"]
metrics = [hierarchy_2Levels, hierarchy_2Levels_noAug, hierarchy_2Levels_rotAug, hierarchy_2Levels_noSkip, hierarchy_2Levels_lr00001]
out_path = out_dir/ "pyramid_exp5_tweaks_level2.svg"
draw_metrics(metrics, exp_name, 50, 0, out_path)
