"""Panel renderer: raw Spearman rho scatter with broken x-axis for all label sources."""

import sys

import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
    ARCHITECTURES, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_arch_data, fetch_baseline, fetch_baseline_ci, format_yaxis,
)

sys.path.insert(0, "manuscript/figures")
from fig_utils import BREAK_1K_POS, draw_xaxis_break


def _format_broken_xaxis(ax, show_xlabel):
    """Log-2 x-axis with broken gap before 1000-way position."""
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: label_map.get(int(round(val)), "")))
        ax.set_xlabel("Granularity", fontsize=9, labelpad=4)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=10)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def plot_raw(ax, dataset, region, show_ylabel=True, show_xlabel=True,
             show_untrained_label=False, tick_interval=None):
    """Raw Spearman rho scatter (all architectures) + 1000-way marker + broken axis."""
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    untrained_mean = fetch_baseline(dataset, region, epoch=0)

    all_y = [bl_mean]
    if not np.isnan(bl_ci_low):
        all_y.append(bl_ci_low)
    if not np.isnan(bl_ci_high):
        all_y.append(bl_ci_high)
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # Coarse conditions (2–64)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = fetch_arch_data(dataset, folder, region)
        for i, m in enumerate(means):
            if not np.isnan(m):
                all_y.extend([m - errs_lo[i], m + errs_hi[i]])
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

    # 1000-way baseline: orange diamond at broken-axis position
    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    # Dashed reference line at 1000-way level (stops at the diamond)
    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    # Untrained baseline
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.25, alpha=0.6, zorder=2)
        if show_untrained_label:
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=8, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax, tick_interval=tick_interval)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)
