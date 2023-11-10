import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


n_epochs = 200
epoch = np.arange(n_epochs)
lr_init = (np.ones(n_epochs) * 1e-3)
lr_diffusionnet = lr_init * (np.ones(n_epochs) * 0.5**np.repeat(np.arange(4),n_epochs/4))
lr_vmnet = lr_init * (1 - epoch / n_epochs)**0.9
lr_picassonet2 = lr_init * 0.98**epoch

plt.plot(epoch, lr_diffusionnet, label="DiffusionNet")
plt.plot(epoch, lr_vmnet, label="VMNet")
plt.plot(epoch, lr_picassonet2, label="PicassoNet++")
plt.legend()
plt.show()



baseline_dir = Path("/home/cychen/Documents/gdl-scene-segment/experiments_baseline")
baseline = "DiffusionNet"
baseline_train = np.loadtxt(baseline_dir/"scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
baseline_test = np.loadtxt(baseline_dir/"scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
baseline_loss = np.loadtxt(baseline_dir/"scannet_semseg_xyz_loss.csv", delimiter=",")



a = np.arange(50)
exp1_dir = baseline_dir/"lr_schedules"

exp1_1 = "room_50_1e-3_PNet"
exp1_2 = "room_50_1e-3_fixed"
exp1_3 = "hal_50_1e-3_VMNet"
name1_1 = "PicassoNet++"
name1_2 = "fixed"
name1_3 = "VMNet"

exp1_1_train = np.loadtxt(exp1_dir/f"{exp1_1}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp1_2_train = np.loadtxt(exp1_dir/f"{exp1_2}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp1_2_train = np.pad(exp1_2_train[:,0], (0,25), mode='edge')
exp1_3_train = np.loadtxt(exp1_dir/f"{exp1_3}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp1_1_test = np.loadtxt(exp1_dir/f"{exp1_1}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp1_2_test = np.loadtxt(exp1_dir/f"{exp1_2}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp1_2_test = np.pad(exp1_2_test[:,0], (0,25), mode='edge')
exp1_3_test = np.loadtxt(exp1_dir/f"{exp1_3}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp1_3_loss = np.loadtxt(exp1_dir/f"{exp1_3}/scannet_semseg_xyz_loss.csv", delimiter=",")

plt.plot(a, baseline_train[:,0], label = f"{baseline}_train")
plt.plot(a, exp1_1_train[:,0], label = f"{name1_1}_train")
plt.plot(a, exp1_2_train, label = f"{name1_2}_train")
plt.plot(a, exp1_3_train[:,0], label = f"{name1_3}_train")
plt.plot(a, baseline_test[:,0], label = f"{baseline}_test")
plt.plot(a, exp1_1_test[:,0], label = f"{name1_1}_test")
plt.plot(a, exp1_2_test, label = f"{name1_2}_test")
plt.plot(a, exp1_3_test[:,0], label = f"{name1_3}_test")
plt.legend()
plt.show()

plt.plot(a, baseline_loss[:,0]/1156, label = f"{baseline}_train")
plt.plot(a, exp1_3_loss[:,0]/1201, label = f"{name1_3}_train")
plt.plot(a, baseline_loss[:,1]/312, label = f"{baseline}_test")
plt.plot(a, exp1_3_loss[:,1]/312, label = f"{name1_3}_test")
plt.legend()
plt.show()



exp2_dir = baseline_dir/"k_eig"

exp2_2 = "hal_50_1e-3_keig64_DiffNet_13"
exp2_3 = "room_50_1e-3_keig32_DiffNet_13"
name2_1 = "k_eig_128"
name2_2 = "k_eig_64"
name2_3 = "k_eig_32"

exp2_2_train = np.loadtxt(exp2_dir/f"{exp2_2}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp2_3_train = np.loadtxt(exp2_dir/f"{exp2_3}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp2_2_test = np.loadtxt(exp2_dir/f"{exp2_2}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp2_3_test = np.loadtxt(exp2_dir/f"{exp2_3}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)

exp2_2_loss = np.loadtxt(exp2_dir/f"{exp2_2}/scannet_semseg_xyz_loss.csv", delimiter=",")
exp2_3_loss = np.loadtxt(exp2_dir/f"{exp2_3}/scannet_semseg_xyz_loss.csv", delimiter=",")

plt.plot(a, baseline_train[:,0], label = f"{name2_1}_train")
plt.plot(a, exp2_2_train[:,0], label = f"{name2_2}_train")
plt.plot(a, exp2_3_train[:,0], label = f"{name2_3}_train")
plt.plot(a, baseline_test[:,0], label = f"{name2_1}_test")
plt.plot(a, exp2_2_test[:,0], label = f"{name2_2}_test")
plt.plot(a, exp2_3_test[:,0], label = f"{name2_3}_test")
plt.legend()
plt.show()

plt.plot(a, baseline_loss[:,0]/1156, label = f"{name2_1}_train")
plt.plot(a, exp2_2_loss[:,0]/1201, label = f"{name2_2}_train")
plt.plot(a, exp2_3_loss[:,0]/1156, label = f"{name2_3}_train")
plt.plot(a, baseline_loss[:,1]/312, label = f"{name2_1}_test")
plt.plot(a, exp2_2_loss[:,1]/312, label = f"{name2_2}_test")
plt.plot(a, exp2_3_loss[:,1]/312, label = f"{name2_3}_test")
plt.legend()
plt.show()



b = np.arange(25)
exp3_dir = baseline_dir/"color_gradient_rotation"

exp3_2 = "hal_wo_grad_roto"
exp3_3 = "room_rgb"
exp3_4 = "hal_rgb_wo_grad_roto"
name3_1 = "xyz with gradient rotation"
name3_2 = "xyz without gradient rotation"
name3_3 = "xyzrgb with with gradient rotation"
name3_4 = "xyzrgb without with gradient rotation"

exp3_2_train = np.loadtxt(exp3_dir/f"{exp3_2}/scannet_semseg_xyz_train_ious.csv", delimiter=",", skiprows=1)
exp3_3_train = np.loadtxt(exp3_dir/f"{exp3_3}/scannet_semseg_xyzrgb_train_ious.csv", delimiter=",", skiprows=1)
exp3_4_train = np.loadtxt(exp3_dir/f"{exp3_4}/scannet_semseg_xyzrgb_train_ious.csv", delimiter=",", skiprows=1)
exp3_2_test = np.loadtxt(exp3_dir/f"{exp3_2}/scannet_semseg_xyz_test_ious.csv", delimiter=",", skiprows=1)
exp3_3_test = np.loadtxt(exp3_dir/f"{exp3_3}/scannet_semseg_xyzrgb_test_ious.csv", delimiter=",", skiprows=1)
exp3_4_test = np.loadtxt(exp3_dir/f"{exp3_4}/scannet_semseg_xyzrgb_test_ious.csv", delimiter=",", skiprows=1)

exp3_2_loss = np.loadtxt(exp3_dir/f"{exp3_2}/scannet_semseg_xyz_loss.csv", delimiter=",")
exp3_3_loss = np.loadtxt(exp3_dir/f"{exp3_3}/scannet_semseg_xyzrgb_loss.csv", delimiter=",")
exp3_4_loss = np.loadtxt(exp3_dir/f"{exp3_4}/scannet_semseg_xyzrgb_loss.csv", delimiter=",")

plt.plot(b, baseline_train[:25,0], label = f"{name3_1}_train")
plt.plot(b, exp3_2_train[:,0], label = f"{name3_2}_train")
plt.plot(b, exp3_3_train[:,0], label = f"{name3_3}_train")
plt.plot(b, exp3_4_train[:,0], label = f"{name3_4}_train")
plt.plot(b, baseline_test[:25,0], label = f"{name3_1}_test")
plt.plot(b, exp3_2_test[:,0], label = f"{name3_2}_test")
plt.plot(b, exp3_3_test[:,0], label = f"{name3_3}_test")
plt.plot(b, exp3_4_test[:,0], label = f"{name3_4}_test")
plt.legend()
plt.show()

plt.plot(b, baseline_loss[:25,0]/1156, label = f"{name3_1}_train")
plt.plot(b, exp3_2_loss[:,0]/1201, label = f"{name3_2}_train")
plt.plot(b, exp3_3_loss[:,0]/1156, label = f"{name3_3}_train")
plt.plot(b, exp3_4_loss[:,0]/1201, label = f"{name3_4}_train")
plt.plot(b, baseline_loss[:25,1]/312, label = f"{name3_1}_test")
plt.plot(b, exp3_2_loss[:,1]/312, label = f"{name3_2}_test")
plt.plot(b, exp3_3_loss[:,1]/312, label = f"{name3_3}_test")
plt.plot(b, exp3_4_loss[:,1]/312, label = f"{name3_4}_test")
plt.legend()
plt.show()



c = np.arange(100)
exp4_dir = baseline_dir/"batchsize_color_augment"

# exp4_2 = "room_100_50_16_rgb_allAugment"
exp4_3 = "hal_100_50_16_rgb_allAugment"
name4_1 = "size1 rotation"
# name4_2 = "room size16 color all augment"
name4_3 = "hal size16 color all augment"

# exp4_2_train = np.loadtxt(exp4_dir/f"{exp4_2}/scannet_semseg_xyzrgb_train_ious.csv", delimiter=",", skiprows=1)
# exp4_2_train = np.pad(exp4_2_train[:,0], (0,67), mode='edge')
exp4_3_train = np.loadtxt(exp4_dir/f"{exp4_3}/scannet_semseg_xyzrgb_train_ious.csv", delimiter=",", skiprows=1)
baseline_train = np.pad(baseline_train[:,0], (0,50), mode='edge')

# exp4_2_test = np.loadtxt(exp4_dir/f"{exp4_2}/scannet_semseg_xyzrgb_test_ious.csv", delimiter=",", skiprows=1)
# exp4_2_test = np.pad(exp4_2_test[:,0], (0,67), mode='edge')
exp4_3_test = np.loadtxt(exp4_dir/f"{exp4_3}/scannet_semseg_xyzrgb_test_ious.csv", delimiter=",", skiprows=1)
baseline_test = np.pad(baseline_test[:,0], (0,50), mode='edge')

# exp4_2_loss = np.loadtxt(exp4_dir/f"{exp4_2}/scannet_semseg_xyzrgb_loss.csv", delimiter=",")
# exp4_2_loss = np.pad(exp4_2_loss, ((0,67), (0,0)), mode='edge')
exp4_3_loss = np.loadtxt(exp4_dir/f"{exp4_3}/scannet_semseg_xyzrgb_loss.csv", delimiter=",")
baseline_loss = np.pad(baseline_loss, ((0,50), (0,0)), mode='edge')

plt.plot(c, baseline_train, label = f"{name4_1}_train")
# plt.plot(c, exp4_2_train, label = f"{name4_2}_train")
plt.plot(c, exp4_3_train[:,0], label = f"{name4_3}_train")
plt.plot(c, baseline_test, label = f"{name4_1}_test")
# plt.plot(c, exp4_2_test, label = f"{name4_2}_test")
plt.plot(c, exp4_3_test[:,0], label = f"{name4_3}_test")
plt.legend()
plt.show()

plt.plot(c, baseline_loss[:,0]/1156, label = f"{name4_1}_train")
# plt.plot(c, exp4_2_loss[:,0], label = f"{name4_2}_train")
plt.plot(c, exp4_3_loss[:,0], label = f"{name4_3}_train")
plt.plot(c, baseline_loss[:,1]/312, label = f"{name4_1}_test")
# plt.plot(c, exp4_2_loss[:,1], label = f"{name4_2}_test")
plt.plot(c, exp4_3_loss[:,1], label = f"{name4_3}_test")
plt.legend()
plt.show()




n_epochs = 50
d = np.arange(n_epochs)
exp5_dir = baseline_dir/"class20_batchsize"

# exp5_2 = "room_25_50_16_rgb_allAug"
exp5_3 = "room_50_VMNet_16_rgb_allAug_20classes"
exp5_4 = "hal_50_VMNet_8_rgb_allAug_20classes"
exp5_5 = "hal_50_VMNet_4_rgb_allAug_20classes"
name5_1 = "epo50_step13_bs1_NoAug_class21_xyz"
# name5_2 = "epo25_step50_bs16_allAug_class21_rgb"
name5_3 = "epo50_VMNet_bs16_allAug_class20_rgb"
name5_4 = "epo50_VMNet_bs8_allAug_class20_rgb"
name5_5 = "epo50_VMNet_bs4_allAug_class20_rgb"

# exp5_2_train = np.loadtxt(exp5_dir/f"{exp5_2}/scannet_semseg_train_ious.csv", delimiter=",", skiprows=1)
exp5_3_train = np.loadtxt(exp5_dir/f"{exp5_3}/scannet_semseg_train_ious.csv", delimiter=",", skiprows=1)
exp5_4_train = np.loadtxt(exp5_dir/f"{exp5_4}/scannet_semseg_train_ious.csv", delimiter=",", skiprows=1)
exp5_5_train = np.loadtxt(exp5_dir/f"{exp5_5}/scannet_semseg_train_ious.csv", delimiter=",", skiprows=1)
# exp5_2_train = np.pad(exp5_2_train, ((0, n_epochs-exp5_2_train.shape[0]),(0,0)), mode='edge')
exp5_4_train = np.pad(exp5_4_train, ((0, n_epochs-exp5_4_train.shape[0]),(0,0)), mode='edge')
exp5_5_train = np.pad(exp5_5_train, ((0, n_epochs-exp5_5_train.shape[0]),(0,0)), mode='edge')
baseline_train = np.pad(baseline_train, ((0, n_epochs-baseline_train.shape[0]),(0,0)), mode='edge')

# exp5_2_test = np.loadtxt(exp5_dir/f"{exp5_2}/scannet_semseg_test_ious.csv", delimiter=",", skiprows=1)
exp5_3_test = np.loadtxt(exp5_dir/f"{exp5_3}/scannet_semseg_test_ious.csv", delimiter=",", skiprows=1)
exp5_4_test = np.loadtxt(exp5_dir/f"{exp5_4}/scannet_semseg_test_ious.csv", delimiter=",", skiprows=1)
exp5_5_test = np.loadtxt(exp5_dir/f"{exp5_5}/scannet_semseg_test_ious.csv", delimiter=",", skiprows=1)
# exp5_2_test = np.pad(exp5_2_test, ((0, n_epochs-exp5_2_test.shape[0]),(0,0)), mode='edge')
exp5_4_test = np.pad(exp5_4_test, ((0, n_epochs-exp5_4_test.shape[0]),(0,0)), mode='edge')
exp5_5_test = np.pad(exp5_5_test, ((0, n_epochs-exp5_5_test.shape[0]),(0,0)), mode='edge')
baseline_test = np.pad(baseline_test, ((0, n_epochs-baseline_test.shape[0]),(0,0)), mode='edge')

# exp5_2_loss = np.loadtxt(exp5_dir/f"{exp5_2}/scannet_semseg_loss.csv", delimiter=",")
exp5_3_loss = np.loadtxt(exp5_dir/f"{exp5_3}/scannet_semseg_loss.csv", delimiter=",")
exp5_4_loss = np.loadtxt(exp5_dir/f"{exp5_4}/scannet_semseg_loss.csv", delimiter=",")
exp5_5_loss = np.loadtxt(exp5_dir/f"{exp5_5}/scannet_semseg_loss.csv", delimiter=",")
# exp5_2_loss = np.pad(exp5_2_loss, ((0, n_epochs-exp5_2_loss.shape[0]),(0,0)), mode='edge')
exp5_4_loss = np.pad(exp5_4_loss, ((0, n_epochs-exp5_4_loss.shape[0]),(0,0)), mode='edge')
exp5_5_loss = np.pad(exp5_5_loss, ((0, n_epochs-exp5_5_loss.shape[0]),(0,0)), mode='edge')
baseline_loss = np.pad(baseline_loss, ((0,n_epochs-baseline_loss.shape[0]), (0,0)), mode='edge')

plt.plot(d, baseline_train[:,0], label = f"{name5_1}_train")
# plt.plot(d, exp5_2_train[:,0], label = f"{name5_2}_train")
plt.plot(d, exp5_3_train[:,0], label = f"{name5_3}_train")
plt.plot(d, exp5_4_train[:,0], label = f"{name5_4}_train")
plt.plot(d, exp5_5_train[:,0], label = f"{name5_5}_train")
plt.plot(d, baseline_test[:,0], label = f"{name5_1}_test")
# plt.plot(d, exp5_2_test[:,0], label = f"{name5_2}_test")
plt.plot(d, exp5_3_test[:,0], label = f"{name5_3}_test")
plt.plot(d, exp5_4_test[:,0], label = f"{name5_4}_test")
plt.plot(d, exp5_5_test[:,0], label = f"{name5_5}_test")
plt.legend()
plt.show()

plt.plot(d, baseline_loss[:,0]/1156, label = f"{name5_1}_train")
# plt.plot(d, exp5_2_loss[:,0], label = f"{name5_2}_train")
plt.plot(d, exp5_3_loss[:,0], label = f"{name5_3}_train")
plt.plot(d, exp5_4_loss[:,0], label = f"{name5_4}_train")
plt.plot(d, exp5_5_loss[:,0], label = f"{name5_5}_train")
plt.plot(d, baseline_loss[:,1]/312, label = f"{name5_1}_test")
# plt.plot(d, exp5_2_loss[:,1], label = f"{name5_2}_test")
plt.plot(d, exp5_3_loss[:,1], label = f"{name5_3}_test")
plt.plot(d, exp5_4_loss[:,1], label = f"{name5_4}_test")
plt.plot(d, exp5_5_loss[:,1], label = f"{name5_5}_test")
plt.legend()
plt.show()
