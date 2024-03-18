import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path



tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
exp_dir = Path("experiments")

# # Read from losses and ious and save metrics
# exp_name = "hierarchy/08_hierarchy_resLinear"
# loss = np.loadtxt(exp_dir/ exp_name/ "loss.csv", delimiter=',')
# train = np.loadtxt(exp_dir/ exp_name/ "train_iou.csv", delimiter=',', skiprows=1)
# val = np.loadtxt(exp_dir/ exp_name/ "val_iou.csv", delimiter=',', skiprows=1)
# metrics = np.hstack((loss, train[:,0:1], val[:,0:1]))
# header = "Train Loss,Val Loss,Train mIoU,Val mIoU"
# np.savetxt(exp_dir/ exp_name/ "metrics.csv", metrics, delimiter=',', fmt='%.4f', header=header, comments="")

centered = np.loadtxt(exp_dir/ "baseline"/ "01_baseline_centered"/ "metrics.csv", delimiter=',', skiprows=1)
single = np.loadtxt(exp_dir/ "baseline"/ "03_baseline_singleComponents"/ "metrics.csv", delimiter=',', skiprows=1)
filled = np.loadtxt(exp_dir/ "baseline"/ "00_baseline_holesFilled"/ "metrics.csv", delimiter=',', skiprows=1)
multi = np.loadtxt(exp_dir/ "baseline"/ "07_baseline_filledMulti"/ "metrics.csv", delimiter=',', skiprows=1)
centered_rot = np.loadtxt(exp_dir/ "baseline"/ "02_baseline_centered_rotAug"/ "metrics.csv", delimiter=',', skiprows=1)
vc002 = np.loadtxt(exp_dir/ "hierarchy"/ "12_hierarchy_1Level"/ "metrics.csv", delimiter=',', skiprows=1)
block8 = np.loadtxt(exp_dir/ "baseline"/ "05_baseline_8DiffBlocks"/ "metrics.csv", delimiter=',', skiprows=1)
hks = np.loadtxt(exp_dir/ "baseline"/ "06_baseline_hks"/ "metrics.csv", delimiter=',', skiprows=1)

hierarchy_filled = np.loadtxt(exp_dir/ "hierarchy"/ "00_hierarchy_holesFilled"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_euc = np.loadtxt(exp_dir/ "hierarchy"/ "02_hierarchy_noGeo"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_geo = np.loadtxt(exp_dir/ "hierarchy"/ "03_hierarchy_noEuc"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_noAug = np.loadtxt(exp_dir/ "hierarchy"/ "01_hierarchy_noAug"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_1B = np.loadtxt(exp_dir/ "hierarchy"/ "04_hierarchy_1DiffBlock"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_4B = np.loadtxt(exp_dir/ "hierarchy"/ "06_hierarchy_4DiffBlocks"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_c32128 = np.loadtxt(exp_dir/ "hierarchy"/ "05_hierarchy_cWidth_32-128"/ "metrics.csv", delimiter=',', skiprows=1)

hierarchy_centered = np.loadtxt(exp_dir/ "hierarchy"/ "07_hierarchy_centered"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_resLinear = np.loadtxt(exp_dir/ "hierarchy"/ "08_hierarchy_resLinear"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_noRes = np.loadtxt(exp_dir/ "hierarchy"/ "09_hierarchy_noRes"/ "metrics.csv", delimiter=',', skiprows=1)

hierarchy_2Levels = np.loadtxt(exp_dir/ "hierarchy"/ "10_hierarchy_2Levels"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_2Levels_noAug = np.loadtxt(exp_dir/ "hierarchy"/ "11_hierarchy_2Levels_noAug"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_2Levels_noSkip = np.loadtxt(exp_dir/ "hierarchy"/ "13_hierarchy_2Levels_noSkip"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_2Levels_rotAug = np.loadtxt(exp_dir/ "hierarchy"/ "14_hierarchy_2Levels_rotAug"/ "metrics.csv", delimiter=',', skiprows=1)


# Preprocess
exp_name = ["Centered", "Single Components", "Holes Filled", "Single Components + Holes Filled"]
metrics = [centered, single, multi, filled]

exp_name = ["Centered", "Single Components", "Holes Filled", "Hierarchy Centered", "Hierarchy Holes Filled"]
metrics = [centered, single, filled, hierarchy_centered, hierarchy_c32128]

# Hierarchy Branches
exp_name = ["DiffusionNet", "Both Branches", "Euclidean", "Geodesic"]
metrics = [centered, hierarchy_filled, hierarchy_euc, hierarchy_geo]

# Augmentation
exp_name = ["DiffusionNet All Augs", "DiffusionNet Rotational Augs", "Hierarchy All Augs", "Hierarchy no Aug"]
metrics = [centered, centered_rot, hierarchy_filled, hierarchy_noAug]

# Hierarchy Number of DiffusionNet Blocks
exp_name = ["Hierarchy 2 DiffBlocks", "Hierarchy 1 DiffBlock", "Hierarchy 4 DiffBlocks"]
metrics = [hierarchy_filled, hierarchy_1B, hierarchy_4B]

# Hierarchy Channel Width
exp_name = ["Hierarchy 64-160", "Hierarchy 32-128"]
metrics = [hierarchy_filled, hierarchy_c32128]

# Hierarchy Residual Connection
exp_name = ["Hierarchy Residual", "Hierarchy Residual+Linear", "Hierarchy no Additional Residual"]
metrics = [hierarchy_centered, hierarchy_resLinear, hierarchy_noRes]

# Hierarchy 2 Levels Tests
exp_name = ["Hierarchy 2 Levels", "Hierarchy 2 Levels no Augmentation", "Hierarchy 2 Levels no Skip Connection", "Hierarchy 2 Levels Rotational Augmentation"]
metrics = [hierarchy_2Levels, hierarchy_2Levels_noAug, hierarchy_2Levels_noSkip, hierarchy_2Levels_rotAug]

# Recent
exp_name = ["Centered Raw", "Centered VC002", "8 DiffusionNet Blocks", "HKS input"]
metrics = [centered, vc002, block8, hks]



for i, m in enumerate(metrics):
    plt.plot(np.arange(m.shape[0]), m[:,0], label = f"{exp_name[i]} Train", color=tableau_colors[i])
    plt.plot(np.arange(m.shape[0]), m[:,1], label = f"{exp_name[i]} Val", color=tableau_colors[i])
plt.legend()
plt.show()

for i, m in enumerate(metrics):
    plt.plot(np.arange(m.shape[0]), m[:,2], label = f"{exp_name[i]} Train", color=tableau_colors[i])
    plt.plot(np.arange(m.shape[0]), m[:,3], label = f"{exp_name[i]} Val", color=tableau_colors[i])
plt.legend()
plt.show()
