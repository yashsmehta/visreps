"""Panel renderer: raw Spearman rho scatter for all label-source architectures."""

import sys

import numpy as np
import seaborn as sns

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
    ARCHITECTURES, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_arch_data, fetch_baseline, format_xaxis, format_yaxis,
)


def plot_raw(ax, dataset, region, show_ylabel=True, show_xlabel=True,
             show_untrained_label=False):
    """Raw Spearman rho scatter (all architectures) + 1000-way and untrained lines."""
    bl_mean = fetch_baseline(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    untrained_mean = fetch_baseline(dataset, region, epoch=0)

    all_y = [bl_mean]
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = fetch_arch_data(dataset, folder, region)
        all_y.extend(m for m in means if not np.isnan(m))
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))

        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=1.5, capthick=0.5,
                        ecolor=style["color"], elinewidth=0.7, zorder=4)

    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=2)

    # Untrained baseline
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.1, alpha=0.7, zorder=2)
        if show_untrained_label:
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=6.5, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min

    format_xaxis(ax, show_xlabel)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)
