"""Supplementary Figure S3B: Per-Layer Behavioral Alignment (CLIP labels, CustomCNN).

Full per-layer RSA profiles for all 7 granularity levels + untrained,
averaged across seeds. THINGS behavioral similarity (no subjects/regions).

Layout: Single panel with legend on the right.

Usage:
    python manuscript/figures/supplementary/supp_s3b_per_layer_behavioral.py
"""

import sys
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, plot_per_layer_panel, build_per_layer_legend, GRAN_CFGS,
)

OUTPUT = "manuscript/figures/supplementary/S4_per_layer/S4b_per_layer_behavioral.png"
PCA_FOLDER = "pca_labels_clip"


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    fig.subplots_adjust(right=0.78)

    plot_per_layer_panel(ax, "things-behavior", "N/A", pca_folder=PCA_FOLDER,
                         title="THINGS Behavioral", show_ylabel=True,
                         show_xlabel=True, gran_levels=GRAN_CFGS)
    for line in ax.get_lines():
        if line.get_marker() and line.get_marker() != 'None':
            line.set_markersize(7)

    # Legend on the right side
    handles = build_per_layer_legend(GRAN_CFGS)
    fig.legend(handles=handles, loc="center right", fontsize=9,
               frameon=True, fancybox=False, framealpha=0.92,
               edgecolor="#dddddd", borderpad=0.8,
               handletextpad=0.4, labelspacing=0.5,
               title="Granularity", title_fontsize=9.5,
               bbox_to_anchor=(0.98, 0.5))

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
