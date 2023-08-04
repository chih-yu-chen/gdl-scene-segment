import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("/home/cychen/Documents/GDL-scene-segment/pretrained_models")

a = np.arange(50)

exp1 = "room_50_1e-3_PNet"
exp2 = "room_50_1e-3_fixed"
exp3 = "room_50_1e-3_DiffNet_13"
exp4 = "hal_50_1e-3_VMNet"
exp5 = "hal_50_1e-3_keig64_DiffNet_13"
exp6 = "room_50_1e-3_keig32_DiffNet_13"

name1 = "PicassoNet++"
name2 = "fixed"
name3 = "DiffusionNet"
name4 = "VMNet"
name3_2 = "k_eig_128"
name5 = "k_eig_64"
name6 = "k_eig_32"

exp1_train = np.loadtxt(data_dir/f"{exp1}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp2_train = np.loadtxt(data_dir/f"{exp2}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp3_train = np.loadtxt(data_dir/f"{exp3}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp4_train = np.loadtxt(data_dir/f"{exp4}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp5_train = np.loadtxt(data_dir/f"{exp5}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp6_train = np.loadtxt(data_dir/f"{exp6}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)

exp1_test = np.loadtxt(data_dir/f"{exp1}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp2_test = np.loadtxt(data_dir/f"{exp2}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp3_test = np.loadtxt(data_dir/f"{exp3}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp4_test = np.loadtxt(data_dir/f"{exp4}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp5_test = np.loadtxt(data_dir/f"{exp5}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp6_test = np.loadtxt(data_dir/f"{exp6}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)

exp3_loss = np.loadtxt(data_dir/f"{exp3}/scannet_semseg_xyz_loss.csv", delimiter=",")
exp4_loss = np.loadtxt(data_dir/f"{exp4}/scannet_semseg_xyz_loss.csv", delimiter=",")
exp5_loss = np.loadtxt(data_dir/f"{exp5}/scannet_semseg_xyz_loss.csv", delimiter=",")
exp6_loss = np.loadtxt(data_dir/f"{exp6}/scannet_semseg_xyz_loss.csv", delimiter=",")

plt.plot(a, exp1_train[:,0], label = f"{name1}_train")
plt.plot(a, np.pad(exp2_train[:,0], (0,25), mode='edge'), label = f"{name2}_train")
plt.plot(a, exp3_train[:,0], label = f"{name3}_train")
plt.plot(a, exp4_train[:,0], label = f"{name4}_train")
plt.plot(a, exp1_test[:,0], label = f"{name1}_test")
plt.plot(a, np.pad(exp2_test[:,0], (0,25), mode='edge'), label = f"{name2}_test")
plt.plot(a, exp3_test[:,0], label = f"{name3}_test")
plt.plot(a, exp4_test[:,0], label = f"{name4}_test")
plt.legend()
plt.show()

plt.plot(a, exp3_loss[:,0]/1156, label = f"{name3}_train")
plt.plot(a, exp4_loss[:,0]/1201, label = f"{name4}_train")
plt.plot(a, exp3_loss[:,1]/312, label = f"{name3}_test")
plt.plot(a, exp4_loss[:,1]/312, label = f"{name4}_test")
plt.legend()
plt.show()

plt.plot(a, exp3_train[:,0], label = f"{name3_2}_train")
plt.plot(a, exp5_train[:,0], label = f"{name5}_train")
plt.plot(a, exp6_train[:,0], label = f"{name6}_train")
plt.plot(a, exp3_test[:,0], label = f"{name3_2}_test")
plt.plot(a, exp5_test[:,0], label = f"{name5}_test")
plt.plot(a, exp6_test[:,0], label = f"{name6}_test")
plt.legend()
plt.show()

plt.plot(a, exp3_loss[:,0]/1156, label = f"{name3_2}_train")
plt.plot(a, exp5_loss[:,0]/1201, label = f"{name5}_train")
plt.plot(a, exp6_loss[:,0]/1156, label = f"{name6}_train")
plt.plot(a, exp3_loss[:,1]/312, label = f"{name3_2}_test")
plt.plot(a, exp5_loss[:,1]/312, label = f"{name5}_test")
plt.plot(a, exp6_loss[:,1]/312, label = f"{name6}_test")
plt.legend()
plt.show()
