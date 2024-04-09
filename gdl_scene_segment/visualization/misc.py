import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def draw_scannet_hist(out_path):

    vf = np.loadtxt("/media/cychen/HDD/scannet/stats/vertices_faces.txt", delimiter=',', skiprows=1, dtype=np.int_)
    v002 = np.loadtxt("/media/cychen/HDD/scannet/stats/vertices_vc002.txt", delimiter=',', skiprows=1, dtype=np.int_)
    _, ax = plt.subplots(layout="constrained")
    _ = ax.hist(vf[:,0], bins=200, alpha=0.7, label="Raw Scenes")
    _ = ax.hist(v002, bins=200, alpha=0.7, label="Simplified Scenes")
    ax.set_title("Number of Vertices in ScanNet Scenes", fontsize=16)
    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Number of Scenes")
    ax.legend(fontsize=8)
    plt.savefig(out_path)
    plt.close()

out_dir = Path("visualizations", "metrics")
draw_scannet_hist(out_dir/ "scannet_hist.svg")
