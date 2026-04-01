"""Supplementary Figure S1: Alternative PCA Source Models (ViT + DINOv3).

Same visual style as main Figure 3 (raw Spearman rho, broken x-axis).
Generates two separate figures:
  S1A: supp_s1a_neural.png  — 2x2 grid (TVSD | NSD, Early | Higher)
  S1B: supp_s1b_behavioral.png — single THINGS panel

Usage:
    python manuscript/figures/supplementary/supp_s1_alternative_pca.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")
sys.path.insert(0, "manuscript/figures/fig3")
sys.path.insert(0, "plotters")

from fig_utils import (
    setup_style, COARSE_CFGS, BREAK_1K_POS, draw_xaxis_break,
    MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
)
from shared import fetch_arch_data, fetch_baseline, fetch_baseline_ci, format_yaxis
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from plotter_utils import get_condition_summary

OUTPUT_DIR = "manuscript/figures/supplementary"

# ViT + DINOv3 — refined color scheme with strong contrast
ARCHITECTURES = [
    ("vit",  "pca_labels_vit",  "ViT"),
    ("dino", "pca_labels_dino", "DINOv3"),
]
ARCH_STYLE = {
    "vit":  {"color": "#c0392b", "marker": "^"},   # vivid crimson
    "dino": {"color": "#1a8a7a", "marker": "o"},    # rich teal
}
BASELINE_1K_COLOR = "#e6850d"


def _format_broken_xaxis(ax, show_xlabel):
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    # Always show tick labels for readability
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: label_map.get(int(round(val)), "")))
    if show_xlabel:
        ax.set_xlabel("Number of classes", fontsize=9, labelpad=5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=7.5)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def plot_panel(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Raw Spearman rho scatter for ViT + DINOv3, Figure 3 style."""
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#999")
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

        # Collect valid points for connecting line
        valid_x, valid_y = [], []
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            x_pos = cfg * jitter
            valid_x.append(x_pos)
            valid_y.append(means[i])
            ax.errorbar(x_pos, means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE + 1,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=0, capthick=0,
                        ecolor=style["color"], elinewidth=0.7, alpha=0.7,
                        zorder=4)

        # Subtle connecting line to show trend
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_y, color=style["color"],
                    linewidth=0.8, alpha=0.3, zorder=3)

    # 1000-way diamond — slightly larger for emphasis
    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE + 1.5,
                markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
                capsize=0, capthick=0,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.8, alpha=0.9,
                zorder=5)

    # Baseline reference lines — subtle
    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=0.8, alpha=0.45, zorder=1)

    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#BBBBBB", linestyle=":",
                    linewidth=0.8, alpha=0.5, zorder=1)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.12)
    format_yaxis(ax)

    # Subtle horizontal gridlines
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=6)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=4)


def _build_legend():
    """Legend with only ViT + DINOv3 (no 1000-way)."""
    handles = [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                      markerfacecolor=ARCH_STYLE[k]["color"],
                      markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                      markersize=6, label=d)
               for k, _, d in ARCHITECTURES]
    return handles


def _fetch_things_data(folder):
    means, errs_lo, errs_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo, ci_hi = s["ci_low"], s["ci_high"]
        errs_lo.append(max(s["mean"] - ci_lo, 0) if not np.isnan(ci_lo) else 0)
        errs_hi.append(max(ci_hi - s["mean"], 0) if not np.isnan(ci_hi) else 0)
    return np.array(means), np.array(errs_lo), np.array(errs_hi)


def _fetch_things_baseline(epoch=20):
    s = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


def generate_s1a():
    """S1A: Neural data — 2x2 grid matching Figure 3 layout."""
    fig = plt.figure(figsize=(8.5, 6.8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.35, wspace=0.30,
                           left=0.15, right=0.96, top=0.87, bottom=0.10)

    panels = [
        (0, 0, "tvsd", "V1",                    True,  False),
        (0, 1, "nsd",  "early visual stream",   False, False),
        (1, 0, "tvsd", "IT",                    True,  True),
        (1, 1, "nsd",  "ventral visual stream", False, True),
    ]

    axes = {}
    for row, col, ds, region, ylabel, xlabel in panels:
        ax = fig.add_subplot(gs[row, col])
        plot_panel(ax, ds, region, show_ylabel=ylabel, show_xlabel=xlabel)
        axes[(row, col)] = ax

    # Region titles — placed as axis titles with clean formatting
    region_titles = {
        (0, 0): "V1", (0, 1): "Early visual stream",
        (1, 0): "IT", (1, 1): "Ventral visual stream",
    }
    for key, label in region_titles.items():
        axes[key].set_title(label, fontsize=9.5, fontweight="medium",
                            color="#333333", pad=7)

    # Column headers — dataset labels above region titles
    for col, title in [(0, "TVSD (Macaque)"), (1, "NSD (Human fMRI)")]:
        pos = axes[(0, col)].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.05, title,
                 fontsize=11, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom")

    # Row labels — rotated, outside left margin with enough clearance
    for row, label in [(0, "Early Visual\nCortex"), (1, "Higher Visual\nCortex")]:
        pos = axes[(row, 0)].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2, label,
                 fontsize=8.5, fontweight="semibold", color="#555555",
                 ha="center", va="center", rotation=90, linespacing=1.4)

    # Vertical separator — subtle dividing line between datasets
    sep_x = (axes[(0, 0)].get_position().x1 + axes[(0, 1)].get_position().x0) / 2
    top_y = axes[(0, 0)].get_position().y1 + 0.03
    bot_y = axes[(1, 0)].get_position().y0 - 0.01
    fig.add_artist(plt.Line2D([sep_x, sep_x], [bot_y, top_y],
                              transform=fig.transFigure, color="#e8e8e8",
                              linewidth=0.7, zorder=0))

    # Legend — inside the NSD early visual stream panel (upper-left, away from data)
    legend_handles = _build_legend()
    legend_handles.append(Line2D([], [], marker="D", color="none",
                                 markerfacecolor=BASELINE_1K_COLOR,
                                 markeredgecolor=EDGE_COLOR,
                                 markeredgewidth=EDGE_WIDTH,
                                 markersize=6, label="1000-way"))
    legend_handles.append(Line2D([0, 1], [0, 0], color=BASELINE_1K_COLOR,
                                 linestyle="--", linewidth=1.0,
                                 alpha=0.55, label="1000-way ref."))
    legend_handles.append(Line2D([0, 1], [0, 0], color="#BBBBBB",
                                 linestyle=":", linewidth=1.0,
                                 alpha=0.6, label="Untrained"))
    axes[(0, 1)].legend(handles=legend_handles, fontsize=7, frameon=True,
                        fancybox=False, framealpha=0.95, edgecolor="#e0e0e0",
                        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                        title="Label source", title_fontsize=7.5,
                        loc="lower left")

    out = f"{OUTPUT_DIR}/supp_s1a_neural.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def generate_s1b():
    """S1B: THINGS behavioral — single panel."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.8))

    bl_mean, bl_ci_low, bl_ci_high = _fetch_things_baseline(epoch=20)
    un_mean, _, _ = _fetch_things_baseline(epoch=0)

    all_y = [bl_mean]
    if not np.isnan(un_mean):
        all_y.append(un_mean)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = _fetch_things_data(folder)
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

    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.6, zorder=2)

    if not np.isnan(un_mean):
        ax.axhline(un_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.0, alpha=0.6, zorder=2)
        ax.text(0.97, un_mean, "Untrained",
                fontsize=6.5, fontstyle="italic", color="#999999",
                ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    _format_broken_xaxis(ax, show_xlabel=True)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    sns.despine(ax=ax, right=True, top=True, offset=3)

    ax.legend(handles=_build_legend(), fontsize=8, frameon=True,
              fancybox=False, framealpha=0.92, edgecolor="#dddddd",
              borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
              title="Coarse label source", title_fontsize=8,
              loc="upper right")

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/supp_s1b_behavioral.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


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
    generate_s1a()
    generate_s1b()


if __name__ == "__main__":
    main()
