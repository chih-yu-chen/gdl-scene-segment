import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path



tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
exp_dir = Path("experiments")

# # Read from losses and ious and save metrics
# exp_name = "hierarchy_centered"
# loss = np.loadtxt(exp_dir/ exp_name/ "loss.csv", delimiter=',')
# train = np.loadtxt(exp_dir/ exp_name/ "train_iou.csv", delimiter=',', skiprows=1)
# val = np.loadtxt(exp_dir/ exp_name/ "val_iou.csv", delimiter=',', skiprows=1)
# metrics = np.hstack((loss, train[:,0:1], val[:,0:1]))
# header = "Train Loss,Val Loss,Train mIoU,Val mIoU"
# np.savetxt(exp_dir/ exp_name/ "metrics.csv", metrics, delimiter=',', fmt='%.4f', header=header, comments="")

centered_metrics = np.loadtxt(exp_dir/ "baseline_centered"/ "metrics.csv", delimiter=',', skiprows=1)
single_metrics = np.loadtxt(exp_dir/ "baseline_SingleComponents"/ "metrics.csv", delimiter=',', skiprows=1)
filled_metrics = np.loadtxt(exp_dir/ "baseline_HolesFilled"/ "metrics.csv", delimiter=',', skiprows=1)
centered_rot_metrics = np.loadtxt(exp_dir/ "baseline_centered_rotAug"/ "metrics.csv", delimiter=',', skiprows=1)
vc002_metrics = np.loadtxt(exp_dir/ "baseline_vc002"/ "metrics.csv", delimiter=',', skiprows=1)

hierarchy_metrics = np.loadtxt(exp_dir/ "test_hierarchy"/ "metrics.csv", delimiter=',', skiprows=1)
euclidean_metrics = np.loadtxt(exp_dir/ "test_no_GeoBranch"/ "metrics.csv", delimiter=',', skiprows=1)
geodesic_metrics = np.loadtxt(exp_dir/ "test_no_EucBranch"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_centered_metrics = np.loadtxt(exp_dir/ "hierarchy_centered"/ "metrics.csv", delimiter=',', skiprows=1)



exp_name = ["DiffusionNet", "Both Branches", "Euclidean", "Geodesic"]
metrics = [centered_metrics, hierarchy_metrics, euclidean_metrics, geodesic_metrics]

exp_name = ["DiffusionNet All Augs", "DiffusionNet Rotational Augs"]
metrics = [centered_metrics, centered_rot_metrics]

exp_name = ["Centered", "Single Components", "Holes Filled"]
metrics = [centered_metrics, single_metrics, filled_metrics]

exp_name = ["DiffusionNet Centered", "Hierarchy Holes Filled", "Hierarchy Centered"]
metrics = [centered_metrics, hierarchy_metrics, hierarchy_centered_metrics]

exp_name = ["Centered Raw", "Centered VC002"]
metrics = [centered_metrics, vc002_metrics]

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
