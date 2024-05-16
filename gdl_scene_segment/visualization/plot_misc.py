import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



# Visualize proportion of vertices after mesh simplification
def draw_nvertices_hist(out_path, centered, level1):

    _, ax = plt.subplots(layout="constrained")
    _ = ax.hist(centered[:,0], bins=200, alpha=0.7, label="Raw Scenes")
    _ = ax.hist(level1[:,0], bins=200, alpha=0.7, label="Simplified Scenes")
    ax.set_title("Number of Vertices in ScanNet Scenes", fontsize=16)
    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Number of Scenes")
    ax.legend(fontsize=8)
    plt.savefig(out_path)
    plt.close()

    return


# Visualize eigenbases on (dis-)connected meshes
def draw_single_comp_proportion(out_path, centered, single):

    _, ax = plt.subplots(layout="constrained")
    _ = ax.hist((single/centered)[:,0], bins=100, alpha=0.9)
    ax.set_title("Largest Connected Component's Proportion", fontsize=16)
    ax.set_xlabel("Proportion of Vertices")
    ax.set_ylabel("Number of Scenes")
    plt.savefig(out_path)
    plt.close()

    return



data_dir = Path("/media/cychen/HDD/scannet")
split_dir = Path(__file__).parent/ "splits"

with open(split_dir/ "scannetv2_train.txt", 'r') as f:
    scenes = f.read().splitlines()
with open(split_dir/ "scannetv2_val.txt", 'r') as f:
    scenes.extend(f.read().splitlines())

centered = np.loadtxt(data_dir/ "stats"/ "nvnf_centered.txt", dtype=np.int_)
l1 = np.loadtxt(data_dir/ "stats"/ "nvnf_level1.txt", dtype=np.int_)
single = np.loadtxt(data_dir/ "stats"/ "nvnf_single_components.txt", dtype=np.int_)

out_dir = Path("visualizations", "eigenbases")
# draw_single_comp_proportion(out_dir/ "largest_comp_proportion.svg", centered, single)

out_dir = Path("visualizations", "metrics")
draw_nvertices_hist(out_dir/ "n_vertices_hist.png", centered, l1)
