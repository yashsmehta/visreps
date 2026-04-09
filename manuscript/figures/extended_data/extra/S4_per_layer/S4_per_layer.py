"""Supplementary Figure S3: Full Per-Layer Granularity Set (All 7 Levels).

Main Figure 3 shows only 3 coarse granularity levels (4, 16, 64) per-layer
for clarity. This supplementary shows ALL 7 levels (2, 4, 8, 16, 32, 64, 1000)
plus untrained.

Layout: 2x3 grid
  Row 0: TVSD V1 | TVSD V4 | TVSD IT
  Row 1: NSD Early | NSD Ventral | THINGS

Usage:
    python manuscript/figures/extended_data/supp_s3_full_per_layer.py
"""

import sys
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, plot_per_layer_panel, build_per_layer_legend, GRAN_CFGS,
)

OUTPUT = "manuscript/figures/extended_data/extra/S4_per_layer/S4_per_layer.png"

PANELS = [
    (0, 0, "tvsd", "V1", "TVSD V1"),
    (0, 1, "tvsd", "V4", "TVSD V4"),
    (0, 2, "tvsd", "IT", "TVSD IT"),
    (1, 0, "nsd", "early visual stream", "NSD Early"),
    (1, 1, "nsd", "ventral visual stream", "NSD Ventral"),
    (1, 2, "things-behavior", "N/A", "THINGS"),
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.32})

    for row, col, dataset, region, title in PANELS:
        ax = axes[row, col]
        show_ylabel = (col == 0)
        show_xlabel = (row == 1)
        plot_per_layer_panel(ax, dataset, region, pca_folder=None,
                             title=title, show_ylabel=show_ylabel,
                             show_xlabel=show_xlabel, gran_levels=GRAN_CFGS)
        for line in ax.get_lines():
            if line.get_marker() and line.get_marker() != 'None':
                line.set_markersize(6.5)

    handles = build_per_layer_legend(GRAN_CFGS)
    fig.legend(handles=handles, loc="lower center", fontsize=10,
               frameon=False, ncol=len(handles),
               handletextpad=0.3, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.01))

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
