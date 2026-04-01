"""Supplementary Figure S6: Neural Reconstruction Analysis.

Reconstruction control: alignment (Spearman rho) vs. number of PCs retained
(top-k) for the best coarse model vs. 1000-way, across all neural
dataset-region pairs.

Layout: 2x3 grid
  Row 0: TVSD V1 | TVSD V4 | TVSD IT
  Row 1: NSD Early | NSD Ventral | (empty)

Usage:
    python manuscript/figures/supplementary/supp_s6_reconstruction.py
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style, plot_reconstruction_panel

OUTPUT = "manuscript/figures/supplementary/supp_s6_reconstruction.png"

# Best coarse model per region: region -> (cfg_id, checkpoint_dir)
RECON_CONFIGS = {
    "tvsd": {
        "V1": (64, "/data/ymehta3/alexnet_pca"),
        "V4": (64, "/data/ymehta3/alexnet_pca"),
        "IT": (64, "/data/ymehta3/alexnet_pca"),
    },
    "nsd": {
        "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
        "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
    },
}

PANELS = [
    (0, 0, "tvsd", "V1",                    "TVSD V1"),
    (0, 1, "tvsd", "V4",                    "TVSD V4"),
    (0, 2, "tvsd", "IT",                    "TVSD IT"),
    (1, 0, "nsd",  "early visual stream",   "NSD Early Visual"),
    (1, 1, "nsd",  "ventral visual stream",  "NSD Ventral Visual"),
]


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    for row, col, dataset, region, title in PANELS:
        ax = fig.add_subplot(gs[row, col])
        show_ylabel = (col == 0)
        config = RECON_CONFIGS[dataset]
        print(f"Plotting reconstruction: {dataset} / {region}")
        plot_reconstruction_panel(ax, dataset, region, title, config,
                                  show_ylabel=show_ylabel)
        ax.legend(fontsize=7, loc="lower right", frameon=True,
                  edgecolor="#dddddd", fancybox=False, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.22, framealpha=0.94)

    # Hide empty bottom-right panel
    ax_empty = fig.add_subplot(gs[1, 2])
    ax_empty.set_visible(False)

    # Panel labels
    panel_idx = 0
    for row, col, _, _, _ in PANELS:
        ax = fig.axes[panel_idx]
        label = chr(ord("a") + panel_idx)
        ax.text(-0.10, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")
        panel_idx += 1

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
