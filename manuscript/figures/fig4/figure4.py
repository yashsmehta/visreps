"""Figure 4: THINGS Behavioral Alignment.

Layout:
  Row 0 (top): [Reconstruction | Coarseness | Scatter | Histogram]
  Row 1 (bottom): [3 RDMs side by side + colorbar] spanning full width

Usage:
    python manuscript/figures/fig4/figure4.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes,
    plot_reconstruction_panel,
)
from things_utils import compute_things_data, plot_rdm_panels, plot_scatter_panel

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig4"

# ── Figure 4 color scheme (same as Figure 3) ──
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue
    "pixels":  {"color": "#c0a898", "marker": "v"},    # muted tan
}
BASELINE_1K_COLOR = "#e8963e"  # warm amber

# THINGS reconstruction config: ViT-PCA 64-way
THINGS_RECON_CONFIG = {"N/A": (64, "/data/ymehta3/vit_pca")}
THINGS_RECON_COLOR = "#08519c"  # CLIP dark blue


# ── Coarseness data fetching ─────────────────────────────────────────────

def fetch_things_arch_data(folder):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_coarseness_raw(ax):
    """Plot raw Spearman ρ coarseness for THINGS behavioral."""
    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]

    # Untrained baseline
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=1)

    # 1000-way horizontal reference line + diamond
    if not np.isnan(bl_mean):
        ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="-",
                    linewidth=0.6, alpha=0.35, zorder=2)
        bl_err_lo = max(bl["mean"] - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
        bl_err_hi = max(bl["ci_high"] - bl["mean"], 0) if not np.isnan(bl["ci_high"]) else 0
        ax.errorbar(BREAK_1K_POS, bl_mean, yerr=[[bl_err_lo], [bl_err_hi]],
                     fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE + 1,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2.5, capthick=0.8, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    # Coarse architectures (AlexNet, CLIP, Pixels — no ViT)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_things_arch_data(folder)
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            ax.errorbar(cfg * jitter, means[i], yerr=[[e_lo], [e_hi]],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE + 1,
                         markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                         capsize=2.5, capthick=0.8,
                         ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_coarseness_axes(ax, "", show_ylabel=True, show_xlabel=True)
    ax.set_title("Alignment vs. Granularity",
                 fontsize=9.5, fontweight="semibold", pad=8)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    fig = plt.figure(figsize=(13, 9))

    # Top row: 4 columns; Bottom row: RDMs spanning full width
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.15],
                           hspace=0.38, left=0.06, right=0.96,
                           top=0.95, bottom=0.04)

    # ── Top row: 4 panels ──
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
                                              wspace=0.48,
                                              width_ratios=[1, 0.85, 1, 0.85])

    # Panel A: Coarseness (Alignment vs. Granularity)
    ax_coarse = fig.add_subplot(gs_top[0, 0])
    plot_coarseness_raw(ax_coarse)

    # Coarseness legend — local color scheme
    legend_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=MARKER_SIZE, label=display)
        legend_handles.append(h)
    legend_handles.append(Line2D([], [], marker="D", color="none",
                                  markerfacecolor=BASELINE_1K_COLOR,
                                  markeredgecolor=EDGE_COLOR,
                                  markeredgewidth=EDGE_WIDTH,
                                  markersize=MARKER_SIZE, label="1K (ImageNet)"))
    legend_handles.append(Line2D([], [], **UNTRAINED_LINE_STYLE, label="Untrained"))

    leg_c = ax_coarse.legend(
        handles=legend_handles,
        fontsize=6, frameon=True, loc="lower left",
        edgecolor="#dddddd", fancybox=False, framealpha=0.95,
        handletextpad=0.25, columnspacing=0.5, ncol=2,
        borderpad=0.3, bbox_to_anchor=(0.0, 0.0))
    leg_c.get_frame().set_linewidth(0.5)

    # Panel B: Reconstruction (THINGS)
    ax_recon = fig.add_subplot(gs_top[0, 1])
    plot_reconstruction_panel(ax_recon, "things-behavior", "N/A",
                              "Reconstruction", THINGS_RECON_CONFIG,
                              show_ylabel=True,
                              coarse_color_override=THINGS_RECON_COLOR)
    ax_recon.legend(fontsize=5.5, loc="lower right", frameon=True,
                    edgecolor="#dddddd", fancybox=False, handletextpad=0.4,
                    borderpad=0.3, labelspacing=0.22, framealpha=0.94)

    # Panels C: Scatter and Histogram
    ax_scatter = fig.add_subplot(gs_top[0, 2])
    ax_hist = fig.add_subplot(gs_top[0, 3])

    # Compute THINGS data (RDMs + scatter data)
    print("Computing THINGS data for scatter and RDMs...")
    precomputed = compute_things_data()

    plot_scatter_panel(ax_scatter, ax_hist, precomputed)

    # ── Bottom row: 3 RDMs + colorbar ──
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1],
                                              wspace=0.06,
                                              width_ratios=[1, 1, 1, 0.05])

    ax_rdm1 = fig.add_subplot(gs_bot[0, 0])
    ax_rdm2 = fig.add_subplot(gs_bot[0, 1])
    ax_rdm3 = fig.add_subplot(gs_bot[0, 2])
    ax_cb = fig.add_subplot(gs_bot[0, 3])

    rdm_axes = [ax_rdm1, ax_rdm2, ax_rdm3]
    plot_rdm_panels(rdm_axes, precomputed, show_difference=False,
                    colorbar_axes=(ax_cb, ax_cb))

    # ── Panel labels ──
    for ax, label, x_off in zip(
        [ax_coarse, ax_recon, ax_scatter, ax_rdm1],
        ["A", "B", "C", "D"],
        [-0.14, -0.12, -0.12, -0.04]):
        ax.text(x_off, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure4.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
