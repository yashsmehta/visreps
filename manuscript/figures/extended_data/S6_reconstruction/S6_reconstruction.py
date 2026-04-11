"""Supplementary Figure S6: Reconstruction Analysis.

Two separate figures:
  (a) Neural: 2x2 grid — TVSD (V1, IT) top row, NSD (Early, Ventral) bottom row
  (b) Behavioral: single THINGS panel

Each panel shows RSA vs number of retained PCs (top-k) for
1000-way (orange) and coarse (unified blue) reconstruction curves,
plus an untrained baseline.

Layout follows S1A (neural 2x2) and S1B (behavioral single panel) structure.

Usage:
    python manuscript/figures/extended_data/supp_s6_reconstruction.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")

from fig_utils import setup_style
from experiments.reconstruction_analysis.plot_utils import (
    query_reconstruction_curve, query_untrained_baseline,
    aggregate_curve,
)

OUTPUT_DIR = "manuscript/figures/extended_data/S6_reconstruction"

# Colors — unified blue for all coarse models, orange for 1000-way
COARSE_COLOR = "#084594"   # 64-way blue (unified for all coarse)
FINE_COLOR = "#FFA500"     # orange — 1000-way
UNTRAINED_COLOR = "#969696"

# Best coarse model per region: region -> (cfg_id, checkpoint_dir)
RECON_CONFIGS = {
    "tvsd": {
        "V1": (64, "/data/ymehta3/alexnet_pca"),
        "IT": (64, "/data/ymehta3/alexnet_pca"),
    },
    "nsd": {
        "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
        "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
    },
    "things-behavior": {
        "N/A": (64, "/data/ymehta3/vit_pca"),
    },
}

# Neural panel definitions
NEURAL_PANELS = [
    (0, 0, "tvsd", "V1",                    True,  False, False),
    (0, 1, "tvsd", "IT",                    False, False, False),
    (1, 0, "nsd",  "early visual stream",   True,  True,  True),
    (1, 1, "nsd",  "ventral visual stream", False, True,  True),
]


def plot_reconstruction_panel(ax, neural_dataset, region,
                              show_ylabel=True, show_xlabel=True,
                              show_untrained_label=False):
    """Draw a single reconstruction panel: coarse vs 1000-way curves."""
    # 1000-way reconstruction curve
    fine_df = query_reconstruction_curve(neural_dataset, region)
    fine_agg = aggregate_curve(fine_df)

    # Coarse reconstruction curve
    cfg_id, checkpoint_dir = RECON_CONFIGS[neural_dataset][region]
    coarse_df = query_reconstruction_curve(
        neural_dataset, region, cfg_id=cfg_id, checkpoint_dir=checkpoint_dir,
    )
    coarse_agg = aggregate_curve(coarse_df)

    # Untrained baseline
    untrained = query_untrained_baseline(neural_dataset, region)

    if fine_agg.empty and coarse_agg.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        return

    # Untrained baseline
    un_mean, un_lo, un_hi = untrained
    if not np.isnan(un_mean):
        if not np.isnan(un_lo):
            ax.axhspan(un_lo, un_hi, color=UNTRAINED_COLOR, alpha=0.08, zorder=0)
        ax.axhline(un_mean, color=UNTRAINED_COLOR, linestyle=":",
                    linewidth=1.3, zorder=1)
        if show_untrained_label:
            ax.text(0.97, un_mean, "Untrained",
                    fontsize=7, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    # 1000-way curve
    if not fine_agg.empty:
        k = fine_agg["pca_k"].values
        ax.fill_between(k, fine_agg["ci_low"].values, fine_agg["ci_high"].values,
                        color=FINE_COLOR, alpha=0.15, zorder=2)
        ax.plot(k, fine_agg["mean"].values, "-o", color=FINE_COLOR, markersize=3,
                linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
                label=r"1000-way (top-$k$)", zorder=3)

    # Coarse curve — unified blue
    if not coarse_agg.empty:
        k_c = coarse_agg["pca_k"].values
        ax.fill_between(k_c, coarse_agg["ci_low"].values, coarse_agg["ci_high"].values,
                        color=COARSE_COLOR, alpha=0.15, zorder=2)
        ax.plot(k_c, coarse_agg["mean"].values, "-s", color=COARSE_COLOR, markersize=3,
                linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
                label=r"Coarse (top-$k$)", zorder=3)

    # Axis formatting
    k_all = fine_agg["pca_k"].values if not fine_agg.empty else coarse_agg["pca_k"].values
    if show_xlabel:
        ax.set_xlabel("Number of PCs ($k$)", fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    ax.set_xticks(k_all)
    labeled = {1, 5, 10, 20, 30, 40, 50} | {int(k_all[0]), int(k_all[-1])}
    ax.set_xticklabels(
        [str(int(v)) if int(v) in labeled else "" for v in k_all], fontsize=7)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.6,
                   direction="out", labelsize=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=2, width=0.4, direction="out")
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=3)


def _build_legend():
    """Legend handles: 1000-way and coarse only (untrained labelled inline)."""
    return [
        Line2D([], [], color=FINE_COLOR, marker="o", markersize=4,
               linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
               label=r"1000-way (top-$k$)"),
        Line2D([], [], color=COARSE_COLOR, marker="s", markersize=4,
               linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
               label=r"Coarse (top-$k$)"),
    ]


def generate_neural():
    """S6A: Neural — 2x2 grid (TVSD top | NSD bottom) x (Early | Higher)."""
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.30, wspace=0.25,
                           left=0.09, right=0.96, top=0.87, bottom=0.10)

    axes = {}
    for row, col, ds, region, ylabel, xlabel, untrained_label in NEURAL_PANELS:
        ax = fig.add_subplot(gs[row, col])
        print(f"Plotting reconstruction: {ds} / {region}")
        plot_reconstruction_panel(ax, ds, region,
                                  show_ylabel=ylabel, show_xlabel=xlabel,
                                  show_untrained_label=untrained_label)
        axes[(row, col)] = ax

    # Region sub-titles
    region_titles = {
        (0, 0): "V1", (0, 1): "IT",
        (1, 0): "Early visual stream", (1, 1): "Ventral visual stream",
    }
    for key, label in region_titles.items():
        axes[key].set_title(label, fontsize=9.5, fontweight="medium",
                            color="#333333", pad=7)

    # Row headers — dataset labels (left side, following S1A pattern)
    fig.canvas.draw()
    for row, title in [(0, "TVSD"), (1, "NSD")]:
        pos = axes[(row, 0)].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2 + 0.015, title,
                 fontsize=11, fontweight="bold", color="#1a1a1a",
                 ha="center", va="center", rotation=90)

    # Column headers — cortical level
    for col, label in [(0, "Early Visual Cortex"), (1, "Higher Visual Cortex")]:
        pos = axes[(0, col)].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.050, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Panel labels (a–d)
    for (row, col), label in zip([(0, 0), (0, 1), (1, 0), (1, 1)], "abcd"):
        pos = axes[(row, col)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend in panel d
    axes[(1, 1)].legend(
        handles=_build_legend(), fontsize=7.5, frameon=True,
        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
        loc="right", bbox_to_anchor=(1.0, 0.45))

    out = f"{OUTPUT_DIR}/S6_reconstruction_neural.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def generate_behavioral():
    """S6B: THINGS behavioral — single reconstruction panel."""
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    fig, ax = plt.subplots(figsize=(5, 3.5))

    print("Plotting reconstruction: things-behavior / N/A")
    plot_reconstruction_panel(ax, "things-behavior", "N/A",
                              show_ylabel=True, show_xlabel=True)

    ax.set_title("THINGS (Behavioral)", fontsize=11, fontweight="bold",
                 color="#333333", pad=8)

    # Legend
    ax.legend(handles=_build_legend(), fontsize=7.5, frameon=True,
              fancybox=False, framealpha=0.92, edgecolor="#dddddd",
              borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
              loc="lower right")

    out = f"{OUTPUT_DIR}/S6_reconstruction_behavioral.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def main():
    generate_neural()
    generate_behavioral()


if __name__ == "__main__":
    main()
