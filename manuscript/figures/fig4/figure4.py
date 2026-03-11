"""Figure 4: Macaque Electrophysiology Alignment (TVSD).

2×3 grid layout (parallel to Figure 3):
  Rows:    V1 (top), IT (bottom)
  Columns: Coarseness (left), Per-Layer (center), Reconstruction (right)

V4 is omitted from the main figure. V4 results in supplementary.

Usage:
    python manuscript/figures/fig4/figure4.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, "plotters")
from plotter_utils import query_best_scores

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
REGIONS = [("V1", "V1"), ("IT", "IT")]
LAYER_PCA_FOLDER = None  # Derived per-region from COARSE_CONFIG below
COARSE_CONFIG = {
    "V1": (64, "/data/ymehta3/alexnet_pca"),
    "IT": (64, "/data/ymehta3/alexnet_pca"),
}
OUTPUT_DIR = "manuscript/figures/fig4"


# ── Column 1: Coarseness ─────────────────────────────────────────────────

def _sem_summary(df):
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
    return mean, sem


def fetch_arch_data(folder, region):
    means, sems = [], []
    for cfg in COARSE_CFGS:
        df = query_best_scores("tvsd", region, folder, cfg,
                               "spearman", epoch=20, analysis="rsa")
        if df.empty:
            means.append(np.nan)
            sems.append(0)
            continue
        m, s = _sem_summary(df)
        means.append(m)
        sems.append(s)
    return np.array(means), np.array(sems)


def plot_coarseness(ax, region, show_ylabel=True, show_xlabel=True):
    un_df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not un_df.empty:
        ax.axhline(un_df.groupby("seed")["score"].mean().mean(),
                    **UNTRAINED_LINE_STYLE, zorder=1)

    bl_df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    if not bl_df.empty:
        bl_mean, bl_sem = _sem_summary(bl_df)
        ax.errorbar(1000, bl_mean, yerr=1.96 * bl_sem,
                     fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES_ALL):
        style = ARCH_STYLE[arch_key]
        means, sems = fetch_arch_data(folder, region)
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES_ALL))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, means[i], yerr=1.96 * sems[i],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE,
                         markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                         capsize=2, capthick=0.7,
                         ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_coarseness_axes(ax, "", show_ylabel=show_ylabel, show_xlabel=show_xlabel)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()

    fig = plt.figure(figsize=(14.0, 7.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                           width_ratios=[1, 1.15, 1],
                           left=0.08, right=0.97, top=0.92, bottom=0.08)

    col_headers = ["Alignment vs. Granularity",
                   "Per-Layer Profile",
                   "Reconstruction Control"]

    axes_grid = []
    layer_arch_names = []
    for row_idx, (region, region_label) in enumerate(REGIONS):
        is_bottom = (row_idx == 1)

        ax_c = fig.add_subplot(gs[row_idx, 0])
        plot_coarseness(ax_c, region, show_ylabel=True, show_xlabel=is_bottom)

        ax_l = fig.add_subplot(gs[row_idx, 1])
        layer_folder = get_layer_folder_from_coarse_config(COARSE_CONFIG, region)
        pca_folder, arch_display = plot_per_layer_panel(
            ax_l, "tvsd", region, layer_folder,
            title=None, show_ylabel=False, show_xlabel=is_bottom)
        layer_arch_names.append(arch_display)

        ax_r = fig.add_subplot(gs[row_idx, 2])
        plot_reconstruction_panel(ax_r, "tvsd", region, "",
                                  COARSE_CONFIG, show_ylabel=False)
        ax_r.set_title("")
        ax_r.set_ylabel("")
        if not is_bottom:
            ax_r.set_xlabel("")
            ax_r.set_xticklabels([])

        axes_grid.append((ax_c, ax_l, ax_r))

    # ── Column headers ──
    for col_idx, header in enumerate(col_headers):
        ax = axes_grid[0][col_idx]
        ax.set_title(header, fontsize=11, fontweight="bold", pad=12,
                     color="#1a1a1a")

    # ── Row labels ──
    for row_idx, (_, region_label) in enumerate(REGIONS):
        ax = axes_grid[row_idx][0]
        ax.annotate(region_label, xy=(-0.30, 0.5), xycoords="axes fraction",
                    fontsize=13, fontweight="bold", rotation=90,
                    ha="center", va="center", color="#222222")

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
    for idx, label in enumerate(labels):
        row, col = divmod(idx, 3)
        ax = axes_grid[row][col]
        ax.text(-0.10, 1.08, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                color="#000000")

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
    out = f"{OUTPUT_DIR}/figure4.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
