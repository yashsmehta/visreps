"""Supplementary Figure S4: Fine-Grained ROI Analysis (NSD).

2x3 grid showing normalized coarseness curves for 6 individual ROIs:
  Top row:    V1, V2, V3
  Bottom row: hV4, FFA, PPA

Each panel: normalized coarseness (% of 1000-way), all 4 architectures
(AlexNet, CLIP, ViT, Pixels).

Usage:
    python manuscript/figures/supplementary/supp_s4_finegrained_roi.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, ".")
from plotters.plotter_utils import get_condition_summary
from manuscript.figures.fig_utils import (
    setup_style, COARSE_CFGS, ARCHITECTURES_ALL, ARCH_STYLE,
    BASELINE_1K_COLOR, UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR,
    EDGE_WIDTH, normalize_to_baseline, format_normalized_coarseness_axes,
    build_coarseness_legend, compute_jitter,
)

OUTPUT = "manuscript/figures/supplementary/supp_s4_finegrained_roi.png"

ROIS = [
    [("V1", "V1"), ("V2", "V2"), ("V3", "V3")],
    [("hV4", "hV4"), ("FFA", "FFA"), ("PPA", "PPA")],
]


def plot_roi_panel(ax, region, region_label, show_ylabel=True, show_xlabel=True):
    """Plot normalized coarseness for one ROI."""
    bl = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    scale = 100.0 / bl_mean

    un = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"] * scale, **UNTRAINED_LINE_STYLE, zorder=1)

    bl_err_lo = max(bl_mean - bl["ci_low"], 0) * scale if not np.isnan(bl["ci_low"]) else 0
    bl_err_hi = max(bl["ci_high"] - bl_mean, 0) * scale if not np.isnan(bl["ci_high"]) else 0
    ax.errorbar(1000, 100.0, yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                elinewidth=1.0, zorder=5)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES_ALL):
        style = ARCH_STYLE[arch_key]
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES_ALL))

        for cfg in COARSE_CFGS:
            s = get_condition_summary("nsd", region, folder, cfg,
                                     "spearman", epoch=20, analysis="rsa")
            if np.isnan(s["mean"]):
                continue
            nm, nlo, nhi = normalize_to_baseline(
                s["mean"], s["ci_low"], s["ci_high"], bl_mean)
            err_lo = max(nm - nlo, 0) if not np.isnan(nlo) else 0
            err_hi = max(nhi - nm, 0) if not np.isnan(nhi) else 0
            ax.errorbar(cfg * jitter, nm, yerr=[[err_lo], [err_hi]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=2, capthick=0.7,
                        ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_normalized_coarseness_axes(ax, region_label,
                                       show_ylabel=show_ylabel,
                                       show_xlabel=show_xlabel)


def main():
    setup_style()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    for row_idx, row_rois in enumerate(ROIS):
        for col_idx, (region, label) in enumerate(row_rois):
            ax = axes[row_idx, col_idx]
            show_ylabel = (col_idx == 0)
            show_xlabel = (row_idx == 1)
            plot_roi_panel(ax, region, label,
                          show_ylabel=show_ylabel, show_xlabel=show_xlabel)

    for i, ax in enumerate(axes.flat):
        label = chr(ord("a") + i)
        ax.text(-0.10, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    handles = build_coarseness_legend(ARCHITECTURES_ALL)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, handletextpad=0.3,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(h_pad=2.0, w_pad=2.5, rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
