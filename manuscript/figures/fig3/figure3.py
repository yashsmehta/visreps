"""Figure 3: Human fMRI Alignment (NSD).

2×3 grid layout:
  Rows:    Early Visual Stream (top), Ventral Visual Stream (bottom)
  Columns: Coarseness (left), Per-Layer (center), Reconstruction (right)

Row labels on the left margin. Column headers at the top. Y-axis labels
only on the leftmost column.

Usage:
    python manuscript/figures/fig3/figure3.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, ARCHITECTURES_ALL, ARCH_STYLE,
    BASELINE_1K_COLOR, UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter, format_coarseness_axes,
    build_coarseness_legend, build_per_layer_legend,
    plot_per_layer_panel, plot_reconstruction_panel,
    get_layer_folder_from_coarse_config,
)

# ── Config ────────────────────────────────────────────────────────────────
REGIONS = [
    ("early visual stream",   "Early Visual\nStream"),
    ("ventral visual stream", "Ventral Visual\nStream"),
]
LAYER_PCA_FOLDER = None  # Derived per-region from COARSE_CONFIG below
COARSE_CONFIG = {
    "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
    "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
}
OUTPUT_DIR = "manuscript/figures/fig3"


# ── Column 1: Coarseness ─────────────────────────────────────────────────

def fetch_arch_data(folder, region):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_coarseness(ax, region, show_ylabel=True, show_xlabel=True):
    un = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=1)

    bl = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_err_lo = max(bl["mean"] - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
    bl_err_hi = max(bl["ci_high"] - bl["mean"], 0) if not np.isnan(bl["ci_high"]) else 0
    if not np.isnan(bl["mean"]):
        ax.errorbar(1000, bl["mean"], yerr=[[bl_err_lo], [bl_err_hi]],
                     fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES_ALL):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_arch_data(folder, region)
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES_ALL))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            ax.errorbar(cfg * jitter, means[i], yerr=[[e_lo], [e_hi]],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE,
                         markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                         capsize=2, capthick=0.7,
                         ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_coarseness_axes(ax, "", show_ylabel=show_ylabel, show_xlabel=show_xlabel)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    fig = plt.figure(figsize=(14.0, 7.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                           width_ratios=[1, 1.15, 1],
                           left=0.09, right=0.97, top=0.92, bottom=0.08)

    col_headers = ["Alignment vs. Granularity",
                   "Per-Layer Profile",
                   "Reconstruction Control"]

    axes_grid = []
    layer_arch_names = []
    for row_idx, (region, region_label) in enumerate(REGIONS):
        is_bottom = (row_idx == 1)

        # Col 1: Coarseness
        ax_c = fig.add_subplot(gs[row_idx, 0])
        plot_coarseness(ax_c, region, show_ylabel=True, show_xlabel=is_bottom)

        # Col 2: Per-layer (uses same architecture as reconstruction control)
        ax_l = fig.add_subplot(gs[row_idx, 1])
        layer_folder = get_layer_folder_from_coarse_config(COARSE_CONFIG, region)
        pca_folder, arch_display = plot_per_layer_panel(
            ax_l, "nsd", region, layer_folder,
            title=None, show_ylabel=False, show_xlabel=is_bottom)
        layer_arch_names.append(arch_display)

        # Col 3: Reconstruction
        ax_r = fig.add_subplot(gs[row_idx, 2])
        plot_reconstruction_panel(ax_r, "nsd", region, "",
                                  COARSE_CONFIG, show_ylabel=False)
        ax_r.set_title("")
        ax_r.set_ylabel("")
        # Normalize spine widths to match other panels
        for spine in ax_r.spines.values():
            spine.set_linewidth(1.0)
        if not is_bottom:
            ax_r.set_xlabel("")
            ax_r.set_xticklabels([])

        axes_grid.append((ax_c, ax_l, ax_r))

    # ── Column headers ──
    for col_idx, header in enumerate(col_headers):
        ax = axes_grid[0][col_idx]
        ax.set_title(header, fontsize=10, fontweight="bold", pad=12,
                     color="#1a1a1a")

    # ── Row labels (region names on left margin) ──
    for row_idx, (_, region_label) in enumerate(REGIONS):
        ax = axes_grid[row_idx][0]
        ax.annotate(region_label, xy=(-0.30, 0.5), xycoords="axes fraction",
                    fontsize=9.5, fontweight="bold", rotation=90,
                    ha="center", va="center", color="#2a2a2a",
                    linespacing=1.1)

    # ── Per-layer architecture annotations (top-left to avoid data overlap) ──
    for row_idx, arch_name in enumerate(layer_arch_names):
        ax = axes_grid[row_idx][1]
        ax.text(0.03, 0.96, arch_name, transform=ax.transAxes,
                fontsize=6.5, ha="left", va="top", color="#888888",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.7))

    # ── Panel labels ──
    labels = ["A", "B", "C", "D", "E", "F"]
    for i, label in enumerate(labels):
        row, col = divmod(i, 3)
        ax = axes_grid[row][col]
        ax.text(-0.10, 1.10, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Remove per-axes legends from col 1 & 2, keep col 3 per-panel ──
    for row in axes_grid:
        for col_idx, ax in enumerate(row):
            legend = ax.get_legend()
            if legend and col_idx != 2:
                legend.remove()

    # ── Reconstruction legends — per-panel (different coarse models per row) ──
    for row_idx in range(len(REGIONS)):
        ax_r = axes_grid[row_idx][2]
        handles, leg_labels = ax_r.get_legend_handles_labels()
        if handles:
            short = [lbl.replace(" (top-$k$ PCs)", "").replace("-way model", "-way")
                     for lbl in leg_labels]
            leg = ax_r.legend(handles, short, fontsize=5.5,
                              loc="lower right", frameon=True,
                              edgecolor="#dddddd", fancybox=False,
                              framealpha=0.95, handletextpad=0.3,
                              borderpad=0.4)
            leg.get_frame().set_linewidth(0.5)

    # ── Shared legends (in gap between rows) ──
    top_bottom = axes_grid[0][0].get_position().y0
    bot_top = axes_grid[1][0].get_position().y1
    mid_y = (top_bottom + bot_top) / 2

    def _col_center(col_idx):
        pos = axes_grid[0][col_idx].get_position()
        return (pos.x0 + pos.x1) / 2

    # Coarseness legend — between rows, aligned with column 1
    leg_c = fig.legend(
        handles=build_coarseness_legend(ARCHITECTURES_ALL),
        loc="center", fontsize=6.5, frameon=True,
        edgecolor="#dddddd", fancybox=False, framealpha=0.95,
        handletextpad=0.3, columnspacing=0.6, ncol=3,
        borderpad=0.5,
        bbox_to_anchor=(_col_center(0), mid_y))
    leg_c.get_frame().set_linewidth(0.5)

    # Per-layer legend — between rows, aligned with column 2
    leg_l = fig.legend(
        handles=build_per_layer_legend(),
        loc="center", fontsize=6.5, frameon=True,
        edgecolor="#dddddd", fancybox=False, framealpha=0.95,
        ncol=4, handletextpad=0.3, columnspacing=0.6,
        borderpad=0.5,
        bbox_to_anchor=(_col_center(1), mid_y))
    leg_l.get_frame().set_linewidth(0.5)

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure3.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
