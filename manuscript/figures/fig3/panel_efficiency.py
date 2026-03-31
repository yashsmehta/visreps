"""Panel renderer: CLIP-only alignment-per-bit efficiency plot."""

import sys

import numpy as np
import seaborn as sns

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_arch_data, fetch_baseline, format_xaxis, format_yaxis,
)


def plot_efficiency(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """CLIP-only alignment-per-bit with connected line + 1000-way reference."""
    bl_mean = fetch_baseline(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    bl_eff = bl_mean / np.log2(1000)
    bits = np.array([np.log2(cfg) for cfg in COARSE_CFGS])

    means, errs_lo, errs_hi = fetch_arch_data(dataset, "pca_labels_clip", region)
    eff_means = means / bits
    errs_lo = errs_lo / bits
    errs_hi = errs_hi / bits

    valid = ~np.isnan(eff_means)
    x_vals = np.array(COARSE_CFGS)[valid]
    y_vals = eff_means[valid]

    clip_color = ARCH_STYLE["clip"]["color"]
    ax.plot(x_vals, y_vals, "-", color=clip_color, linewidth=1.4, zorder=3)
    ax.errorbar(x_vals, y_vals, yerr=[errs_lo[valid], errs_hi[valid]],
                fmt="s", color=clip_color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                capsize=1.5, capthick=0.5,
                ecolor=clip_color, elinewidth=0.6, zorder=4)

    ax.axhline(bl_eff, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=2)

    all_y = np.concatenate([y_vals, [bl_eff]])

    y_range_eff = all_y.max() - all_y.min()
    ax.text(0.03, bl_eff + y_range_eff * 0.04,
            "Trained on\n1000 classes",
            fontsize=6.5, fontstyle="italic", color=BASELINE_1K_COLOR,
            ha="left", va="bottom", linespacing=1.1,
            transform=ax.get_yaxis_transform(), zorder=10)
    y_min, y_max = all_y.min(), all_y.max()
    y_range = y_max - y_min

    format_xaxis(ax, show_xlabel, xlim_right=96)
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.12)
    format_yaxis(ax, fmt_str=".3f")

    if show_ylabel:
        ax.set_ylabel(r"$\rho\, /\, \log_2 K$", fontsize=9, labelpad=6)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=2)
