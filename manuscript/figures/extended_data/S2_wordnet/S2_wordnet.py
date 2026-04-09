"""Supplementary Figure S2: WordNet Hierarchy Labels -- Coarseness Comparison.

Two separate figures:
  (a) Neural: 2x2 scatter grid — TVSD (V1, IT) top row, NSD (Early, Ventral) bottom row
  (b) Behavioral: single THINGS panel

Each panel shows WordNet coarseness levels (2, 3, 4, 10, 20, 57) as scatter points
with broken x-axis, a 1000-way diamond, and untrained baseline — matching Figure 3 style.

Usage:
    python manuscript/figures/extended_data/supp_s2_wordnet.py
"""

import sys
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "plotters")
sys.path.insert(0, "manuscript/figures")

from plotter_utils import get_condition_summary
from fig_utils import (
    setup_style, DB_PATH, BREAK_1K_POS, draw_xaxis_break,
    EDGE_COLOR, EDGE_WIDTH, MARKER_SIZE,
)

OUTPUT_DIR = "manuscript/figures/extended_data/S2_wordnet"

# WordNet config
WORDNET_CFGS = [2, 3, 4, 10, 20, 57]
WORDNET_FOLDER = "pca_labels_wordnet"

# WordNet style — distinct hexagon marker in forest green
WORDNET_COLOR = "#2ca02c"
WORDNET_MARKER = "h"  # hexagon

# Reference styles (matching Figure 3)
BASELINE_1K_COLOR = "#e8963e"

# Neural panel definitions (row, col, dataset, region, panel_title)
NEURAL_PANELS = [
    (0, 0, "tvsd", "V1",                    "V1"),
    (0, 1, "tvsd", "IT",                    "IT"),
    (1, 0, "nsd",  "early visual stream",   "Early Visual Stream"),
    (1, 1, "nsd",  "ventral visual stream", "Ventral Visual Stream"),
]


def get_wordnet_summary(neural_dataset, region, cfg_id):
    """Get mean score (+ cross-subject SEM) for a WordNet condition."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT score FROM results
        WHERE neural_dataset = ? AND region = ? AND cfg_id = ?
          AND pca_labels_folder = ? AND compare_method = 'spearman'
          AND reconstruct_from_pcs = 0 AND epoch = 20
    """, conn, params=[neural_dataset, region, cfg_id, WORDNET_FOLDER])
    conn.close()

    if df.empty:
        return np.nan, 0, 0

    mean = df["score"].mean()
    if len(df) > 1:
        sem = df["score"].std() / np.sqrt(len(df))
        err_lo = 1.96 * sem
        err_hi = 1.96 * sem
    else:
        err_lo, err_hi = 0, 0

    return mean, err_lo, err_hi


def _format_broken_xaxis(ax, show_xlabel):
    """Log-scale x-axis with WordNet ticks + broken gap before 1000."""
    ax.set_xscale("log", base=2)
    all_x = WORDNET_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in WORDNET_CFGS}
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
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def _format_yaxis(ax, tick_interval=None):
    """Y-axis with minor ticks, grid, and trimmed numeric labels."""
    if tick_interval is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))


def plot_wordnet_scatter(ax, neural_dataset, region,
                         show_ylabel=True, show_xlabel=True,
                         show_untrained_label=False):
    """Draw one scatter panel: WordNet points + 1K diamond + untrained baseline."""
    # Fetch 1000-way baseline
    bl = get_condition_summary(
        neural_dataset, region, "imagenet1k", 1000, "spearman", epoch=20, analysis="rsa")
    bl_mean, bl_ci_low, bl_ci_high = bl["mean"], bl["ci_low"], bl["ci_high"]

    # Fetch untrained baseline
    un = get_condition_summary(
        neural_dataset, region, "imagenet1k", 1000, "spearman", epoch=0, analysis="rsa")
    untrained_mean = un["mean"]

    # Collect all y-values for axis limits
    all_y = []
    if not np.isnan(bl_mean):
        all_y.append(bl_mean)
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # WordNet scatter points (2, 3, 4, 10, 20, 57)
    for cfg in WORDNET_CFGS:
        mean, err_lo, err_hi = get_wordnet_summary(neural_dataset, region, cfg)
        if np.isnan(mean):
            continue
        all_y.extend([mean - err_lo, mean + err_hi])
        ax.errorbar(cfg, mean,
                    yerr=[[err_lo], [err_hi]],
                    fmt=WORDNET_MARKER, color=WORDNET_COLOR,
                    markersize=MARKER_SIZE,
                    markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                    capsize=1.5, capthick=0.5,
                    ecolor=WORDNET_COLOR, elinewidth=0.7, zorder=4)

    # 1000-way diamond at break position
    if not np.isnan(bl_mean):
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
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=8, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    # Axis formatting
    if not all_y:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    _format_yaxis(ax)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def plot_neural_figure():
    """Generate the 2x2 neural figure (TVSD top, NSD bottom)."""
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

    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.25, hspace=0.30,
        top=0.88, bottom=0.08, left=0.12, right=0.96,
    )

    axes = {}
    for row, col, nd, region, title in NEURAL_PANELS:
        ax = fig.add_subplot(gs[row, col])
        show_ylabel = (col == 0)
        show_xlabel = (row == 1)
        show_untrained = (row == 1)  # label untrained on bottom row only
        print(f"Drawing: {nd} / {region}")
        plot_wordnet_scatter(ax, nd, region,
                             show_ylabel=show_ylabel,
                             show_xlabel=show_xlabel,
                             show_untrained_label=show_untrained)
        axes[(row, col)] = ax

    # Column headers (cortical level)
    fig.canvas.draw()
    for col, label in [(0, "Early Visual Cortex"), (1, "Higher Visual Cortex")]:
        pos = axes[(0, col)].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.045, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Row headers (dataset + species, left side — two vertically-stacked rotated texts)
    row_headers = {0: ("TVSD", "Macaque"), 1: ("NSD", "Human fMRI")}
    for row_idx, (title, subtitle) in row_headers.items():
        ax = axes[(row_idx, 0)]
        pos = ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        x_label = pos.x0 - 0.078
        fig.text(x_label, y_center + 0.015, title,
                 fontsize=11, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom", rotation=90, fontfamily="sans-serif")
        fig.text(x_label, y_center - 0.015, subtitle,
                 fontsize=8.5, fontweight="normal", color="#777777",
                 fontstyle="italic",
                 ha="center", va="top", rotation=90, fontfamily="sans-serif")

    # Region sub-labels above each panel
    region_labels = {
        (0, 0): "V1",       (0, 1): "IT",
        (1, 0): "Early visual stream", (1, 1): "Ventral visual stream",
    }
    for key, label in region_labels.items():
        pos = axes[key].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, label,
                 fontsize=9, color="#666666", ha="center", va="bottom")

    # Panel labels (a–d)
    panel_labels = [(0, 0, "a"), (0, 1, "b"), (1, 0, "c"), (1, 1, "d")]
    for row, col, label in panel_labels:
        pos = axes[(row, col)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend (inside panel b — just WordNet marker)
    legend_handles = [
        Line2D([], [], marker=WORDNET_MARKER, color="none",
               markerfacecolor=WORDNET_COLOR,
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
               markersize=MARKER_SIZE, label="WordNet labels"),
    ]
    axes[(0, 1)].legend(handles=legend_handles, fontsize=7.5, frameon=True,
                        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                        loc="upper right")

    out = f"{OUTPUT_DIR}/S2_wordnet_neural.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def plot_behavioral_figure():
    """Generate the single-panel THINGS behavioral figure."""
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

    fig, ax = plt.subplots(figsize=(5, 3.5))

    print("Drawing: things-behavior / N/A")
    plot_wordnet_scatter(ax, "things-behavior", "N/A",
                         show_ylabel=True, show_xlabel=True)

    ax.set_title("THINGS (Behavioral)", fontsize=11, fontweight="bold",
                 color="#333333", pad=8)

    # Legend — just WordNet marker
    legend_handles = [
        Line2D([], [], marker=WORDNET_MARKER, color="none",
               markerfacecolor=WORDNET_COLOR,
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
               markersize=MARKER_SIZE, label="WordNet labels"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, frameon=True,
              fancybox=False, framealpha=0.92, edgecolor="#dddddd",
              borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
              loc="upper right")

    out = f"{OUTPUT_DIR}/S2_wordnet_behavioral.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def main():
    plot_neural_figure()
    plot_behavioral_figure()


if __name__ == "__main__":
    main()
