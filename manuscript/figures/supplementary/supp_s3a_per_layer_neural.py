"""Supplementary Figure S3A: Per-Layer Neural Alignment (CLIP labels, CustomCNN).

Full per-layer RSA profiles for all 7 granularity levels + untrained,
averaged across subjects and seeds. Heading style matches S1A.

Layout: 2x2 grid
  Row 0 (TVSD):  V1 | IT
  Row 1 (NSD):   Early visual stream | Ventral visual stream

Usage:
    python manuscript/figures/supplementary/supp_s3a_per_layer_neural.py
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, plot_per_layer_panel, build_per_layer_legend, GRAN_CFGS,
)

OUTPUT = "manuscript/figures/supplementary/supp_s3a_per_layer_neural.png"
PCA_FOLDER = "pca_labels_clip"

PANELS = [
    # (row, col, dataset, region, region_title)
    (0, 0, "tvsd", "V1", "V1"),
    (0, 1, "tvsd", "IT", "IT"),
    (1, 0, "nsd", "early visual stream", "Early visual stream"),
    (1, 1, "nsd", "ventral visual stream", "Ventral visual stream"),
]

ROW_HEADERS = {0: "TVSD", 1: "NSD"}
COL_HEADERS = {0: "Early Visual Cortex", 1: "Higher Visual Cortex"}


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

    fig = plt.figure(figsize=(15.6, 6.8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.30, wspace=0.25,
                           left=0.08, right=0.85, top=0.87, bottom=0.10)

    axes = {}
    for row, col, dataset, region, region_title in PANELS:
        ax = fig.add_subplot(gs[row, col])
        show_ylabel = (col == 0)
        show_xlabel = (row == 1)
        plot_per_layer_panel(ax, dataset, region, pca_folder=PCA_FOLDER,
                             title=region_title, show_ylabel=show_ylabel,
                             show_xlabel=show_xlabel, gran_levels=GRAN_CFGS)
        # Match S1A region title style
        ax.set_title(region_title, fontsize=9.5, fontweight="medium",
                     color="#333333", pad=7)
        for line in ax.get_lines():
            if line.get_marker() and line.get_marker() != 'None':
                line.set_markersize(6.5)
        axes[(row, col)] = ax

    # Row headers — dataset labels (rotated, left of plots)
    for row, title in ROW_HEADERS.items():
        pos = axes[(row, 0)].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2 + 0.015, title,
                 fontsize=11, fontweight="bold", color="#1a1a1a",
                 ha="center", va="center", rotation=90)

    # Column headers — cortical level (above top row)
    for col, label in COL_HEADERS.items():
        pos = axes[(0, col)].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.050, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Panel labels (a, b, c, d)
    for idx, ((row, col), label) in enumerate(zip(
            [(0, 0), (0, 1), (1, 0), (1, 1)], "abcd")):
        pos = axes[(row, col)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend on the right side of the figure
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
