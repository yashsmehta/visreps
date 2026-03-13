"""Figure 5: Per-Concept Alignment Analysis.

Layout:
  [Scatter plot (large)]  [Histogram]

Panel A: Per-concept scatter — CLIP 8-class vs 1000-way per-concept RSA
Panel B: Histogram of per-concept advantage (delta rho)

Usage:
    python manuscript/figures/fig5/figure5.py
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style
from things_utils import compute_things_data, plot_scatter_panel

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig5"


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })

    fig = plt.figure(figsize=(13, 6.2))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                           width_ratios=[1.15, 1.0],
                           left=0.07, right=0.97, top=0.91, bottom=0.11)

    # Panel A: Scatter
    ax_scatter = fig.add_subplot(gs[0, 0])

    # Panel B: Histogram (standalone, not inset)
    ax_hist = fig.add_subplot(gs[0, 1])

    print("Computing THINGS data for per-concept analysis...")
    precomputed = compute_things_data()

    plot_scatter_panel(ax_scatter, ax_hist, precomputed)

    # Override histogram formatting for standalone display
    ax_hist.set_xlabel(
        r"$\Delta\rho_s$ (CLIP 8-class $-$ 1000-class)", fontsize=10.5)
    ax_hist.set_ylabel("Count", fontsize=10.5)
    ax_hist.tick_params(axis="both", labelsize=9, length=4, width=0.8)
    ax_hist.set_title("Per-Concept Advantage", fontsize=12,
                      fontweight="semibold", pad=10)
    sns.despine(ax=ax_hist, offset=5)

    # Panel labels
    for ax, label, x_off in zip(
        [ax_scatter, ax_hist], ["A", "B"], [-0.10, -0.10]):
        ax.text(x_off, 1.08, label, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    out = f"{OUTPUT_DIR}/figure5.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
