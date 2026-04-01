"""Supplementary Figure S5: THINGS Per-Architecture Breakdown.

Main Figure 4 shows all architectures on one panel. This supplementary breaks
it out into 4 individual panels (one per PCA source: AlexNet, CLIP, ViT, Pixels).

Layout: 1x4 grid. Each panel shows normalized coarseness (% of 1000-way) for
a single architecture. X-axis: log2 (2,4,...,64,1000). Y-axis: % of 1000-way.

Usage:
    python manuscript/figures/supplementary/supp_s5_things_architectures.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from plotters.plotter_utils import get_condition_summary
from manuscript.figures.fig_utils import (
    setup_style, COARSE_CFGS, ARCHITECTURES_ALL, ARCH_STYLE,
    BASELINE_1K_COLOR, UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    normalize_to_baseline, format_normalized_coarseness_axes,
)

OUTPUT = "manuscript/figures/supplementary/supp_s5_things_architectures.png"


def plot_single_arch_panel(ax, arch_key, folder, display_name,
                            show_ylabel=True):
    """Plot normalized coarseness for one architecture on THINGS."""
    style = ARCH_STYLE[arch_key]

    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]

    # Untrained
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]) and not np.isnan(bl_mean) and bl_mean != 0:
        un_norm = (un["mean"] / bl_mean) * 100
        ax.axhline(un_norm, **UNTRAINED_LINE_STYLE, zorder=1)

    # 1000-way diamond at 100%
    if not np.isnan(bl_mean):
        bl_err_lo = max(bl["mean"] - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
        bl_err_hi = max(bl["ci_high"] - bl["mean"], 0) if not np.isnan(bl["ci_high"]) else 0
        scale = 100.0 / bl_mean
        ax.errorbar(1000, 100.0, yerr=[[bl_err_lo * scale], [bl_err_hi * scale]],
                     fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    # Coarse levels for this architecture
    x_vals, y_vals, err_lo, err_hi = [], [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        nm, nl, nh = normalize_to_baseline(s["mean"], s["ci_low"],
                                            s["ci_high"], bl_mean)
        if np.isnan(nm):
            continue
        x_vals.append(cfg)
        y_vals.append(nm)
        err_lo.append(max(nm - nl, 0) if not np.isnan(nl) else 0)
        err_hi.append(max(nh - nm, 0) if not np.isnan(nh) else 0)

    if x_vals:
        ax.errorbar(x_vals, y_vals,
                     yerr=[err_lo, err_hi],
                     fmt=f"-{style['marker']}", color=style["color"],
                     markersize=MARKER_SIZE,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2, capthick=0.7,
                     ecolor=style["color"], elinewidth=1.0,
                     linewidth=1.5, zorder=4)

    format_normalized_coarseness_axes(ax, region_label=display_name,
                                       show_ylabel=show_ylabel, show_xlabel=True)


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

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8),
                              gridspec_kw={"wspace": 0.30})

    for i, (arch_key, folder, display) in enumerate(ARCHITECTURES_ALL):
        plot_single_arch_panel(axes[i], arch_key, folder, f"{display} Labels",
                                show_ylabel=(i == 0))

    # Shared legend
    handles = [
        Line2D([], [], marker="D", color="none",
               markerfacecolor=BASELINE_1K_COLOR,
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
               markersize=MARKER_SIZE, label="1K (ImageNet)"),
        Line2D([], [], **UNTRAINED_LINE_STYLE, label="Untrained"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=8,
               frameon=False, ncol=2,
               handletextpad=0.3, columnspacing=1.5,
               bbox_to_anchor=(0.5, -0.04))

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
