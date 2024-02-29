import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path



tableau_colors = list(mcolors.TABLEAU_COLORS.keys())
exp_dir = Path("experiments")

# Read from losses and ious and save metrics
# centered_loss = np.loadtxt(exp_dir/ "baseline_centered"/ "loss.csv", delimiter=',')
# centered_train = np.loadtxt(exp_dir/ "baseline_centered"/ "train_iou.csv", delimiter=',', skiprows=1)
# centered_val = np.loadtxt(exp_dir/ "baseline_centered"/ "val_iou.csv", delimiter=',', skiprows=1)
# centered_metrics = np.hstack((centered_loss, centered_train[:,0:1], centered_val[:,0:1]))
# header = "Train Loss,Val Loss,Train mIoU,Val mIoU"
# np.savetxt(exp_dir/ "baseline_centered"/ "metrics.csv", centered_metrics, delimiter=',', fmt='%.4f', header=header, comments="")

centered_metrics = np.loadtxt(exp_dir/ "baseline_centered"/ "metrics.csv", delimiter=',', skiprows=1)
filled_metrics = np.loadtxt(exp_dir/ "baseline_HolesFilled"/ "metrics.csv", delimiter=',', skiprows=1)
hierarchy_metrics = np.loadtxt(exp_dir/ "test_hierarchy"/ "metrics.csv", delimiter=',', skiprows=1)
euclidean_metrics = np.loadtxt(exp_dir/ "test_no_GeoBranch"/ "metrics.csv", delimiter=',', skiprows=1)
geodesic_metrics = np.loadtxt(exp_dir/ "test_no_EucBranch"/ "metrics.csv", delimiter=',', skiprows=1)
centered_rot_metrics = np.loadtxt(exp_dir/ "baseline_centered_rotAug"/ "metrics.csv", delimiter=',', skiprows=1)



exp_name = ["DiffusionNet", "Both Branches", "Euclidean", "Geodesic"]
metrics = [centered_metrics, hierarchy_metrics, euclidean_metrics, geodesic_metrics]

exp_name = ["DiffusionNet All Augs", "DiffusionNet Rotational Augs"]
metrics = [centered_metrics, centered_rot_metrics]



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
