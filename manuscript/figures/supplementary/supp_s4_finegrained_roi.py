"""Supplementary Figure S4: Fine-Grained ROI Analysis (NSD).

2x3 grid showing raw RSA (Spearman rho) coarseness curves for 6 individual ROIs:
  Top row:    V1, V2, V3
  Bottom row: hV4, FFA, PPA

Same visual style as S1 neural (raw Spearman rho, broken x-axis, jittered scatter,
NO connecting lines). Two PCA label sources: AlexNet, CLIP.
Legend contains only label sources (no 1000-way or untrained entries).

Usage:
    python manuscript/figures/supplementary/supp_s4_finegrained_roi.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")
sys.path.insert(0, "manuscript/figures/fig3")
sys.path.insert(0, "plotters")

from fig_utils import (
    setup_style, COARSE_CFGS, BREAK_1K_POS, draw_xaxis_break,
    MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
)
from shared import fetch_baseline, fetch_baseline_ci, format_yaxis
from plotter_utils import get_condition_summary

OUTPUT = "manuscript/figures/supplementary/supp_s4_finegrained_roi.png"

# ── PCA label sources — same colors as S1 / Figure 3 ────────────────────
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue
}
BASELINE_1K_COLOR = "#e8963e"

ROIS = [
    [("V1", "V1"), ("V2", "V2"), ("V3", "V3")],
    [("hV4", "hV4"), ("FFA", "FFA"), ("PPA", "PPA")],
]


def _fetch_arch_data(folder, region):
    """Fetch means and asymmetric error bars for coarse conditions (2-64)."""
    means, errs_lo, errs_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        m = s["mean"]
        lo = max(m - s["ci_low"], 0) if not np.isnan(s["ci_low"]) else 0
        hi = max(s["ci_high"] - m, 0) if not np.isnan(s["ci_high"]) else 0
        means.append(m)
        errs_lo.append(lo)
        errs_hi.append(hi)
    return np.array(means), np.array(errs_lo), np.array(errs_hi)


def _format_broken_xaxis(ax, show_xlabel):
    """Log-2 x-axis with broken gap before 1000-way position."""
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: label_map.get(int(round(val)), "")))
    if show_xlabel:
        ax.set_xlabel("Granularity", fontsize=9, labelpad=5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def plot_roi_panel(ax, region, region_label, show_ylabel=True, show_xlabel=True,
                   show_untrained_label=False):
    """Raw Spearman rho scatter for one ROI — S1 style."""
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci("nsd", region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#999")
        return

    untrained_mean = fetch_baseline("nsd", region, epoch=0)
    all_y = [bl_mean]
    if not np.isnan(bl_ci_low):
        all_y.append(bl_ci_low)
    if not np.isnan(bl_ci_high):
        all_y.append(bl_ci_high)
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # Coarse conditions (2-64) — jittered per architecture
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = _fetch_arch_data(folder, region)
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

    # Dashed reference line at 1000-way level
    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    # Untrained baseline
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.25, alpha=0.6, zorder=2)
        if show_untrained_label:
            ax.text(0.03, untrained_mean, "Untrained",
                    fontsize=7, fontstyle="italic", color="#999999",
                    ha="left", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax)

    ax.set_title(region_label, fontsize=10, fontweight="semibold", pad=7)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def _build_legend():
    """Legend with only the PCA label sources — no baselines."""
    return [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                   markerfacecolor=ARCH_STYLE[k]["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=6, label=d)
            for k, _, d in ARCHITECTURES]


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig = plt.figure(figsize=(14, 7.5))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.35, wspace=0.25,
                           left=0.07, right=0.97, top=0.92, bottom=0.10)

    axes = {}
    for row_idx, row_rois in enumerate(ROIS):
        for col_idx, (region, label) in enumerate(row_rois):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            show_ylabel = (col_idx == 0)
            show_xlabel = (row_idx == 1)
            # Show "Untrained" label in bottom-left panel only
            show_untrained = (row_idx == 1 and col_idx == 0)
            plot_roi_panel(ax, region, label,
                           show_ylabel=show_ylabel, show_xlabel=show_xlabel,
                           show_untrained_label=show_untrained)
            axes[(row_idx, col_idx)] = ax

    # Panel labels (a-f)
    for idx, (row_idx, col_idx) in enumerate(
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
        label = chr(ord("a") + idx)
        pos = axes[(row_idx, col_idx)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend — label sources only, in panel f (bottom-right)
    axes[(1, 2)].legend(
        handles=_build_legend(), fontsize=7.5, frameon=True,
        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
        title="Coarse label source", title_fontsize=7.5,
        loc="right", bbox_to_anchor=(1.0, 0.30))

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
