"""Figure 4: THINGS Behavioral Alignment.

Layout:
  Row 0 (top): [Schematic | Coarseness | Model Comparison | Data Efficiency]
  Row 1 (bottom): [4 PCA scatter panels] spanning full width

Panel A: Schematic of THINGS behavioral similarity task (placeholder)
Panel B: Alignment vs. Granularity (raw Spearman rho, log x-axis)
Panel C: Model comparison — coarse vs 1000-way bars + pretrained scatter
Panel D: Data efficiency — coarse models on 10K images vs 1000-class
Panel E: PC scatter — Behavioral, CLIP 8-class, Pretrained AlexNet, Pretrained ViT

Usage:
    python manuscript/figures/fig4/figure4.py
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style, draw_schematic_placeholder

sys.path.insert(0, "manuscript/figures/fig4")
from panel_coarseness import plot_coarseness
from panel_comparison import plot_comparison
from panel_data_efficiency import plot_data_efficiency
from panel_scatter import (
    load_pc_scatter_data, draw_image_insets,
    PC_PANELS,
)
from plot_pc_scatter import (
    load_super_categories, compute_pca,
    plot_scatter_panel, SUPER_ORDER, SUPER_COLORS,
)

OUTPUT_DIR = "manuscript/figures/fig4"


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
    })

    fig = plt.figure(figsize=(17.0, 9.0))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           height_ratios=[1.0, 0.88],
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.32, wspace=0.30,
                           left=0.045, right=0.965,
                           top=0.96, bottom=0.06)

    # ── Top row panels ───────────────────────────────────────────────────
    ax_schematic = fig.add_subplot(gs[0, 0])
    draw_schematic_placeholder(ax_schematic,
                               "THINGS\nBehavioral Similarity\n(schematic)")

    ax_coarse = fig.add_subplot(gs[0, 1])
    plot_coarseness(ax_coarse)

    ax_compare = fig.add_subplot(gs[0, 2])
    plot_comparison(ax_compare, ref_ax=ax_coarse)

    ax_data_eff = fig.add_subplot(gs[0, 3])
    plot_data_efficiency(ax_data_eff, ref_ax=ax_coarse)

    # ── Bottom row: 4 PCA scatter panels ─────────────────────────────────
    pc_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    print("Loading PCA scatter data...")
    reps, concept_names = load_pc_scatter_data()
    n_concepts = list(reps.values())[0].shape[0]
    super_labels = load_super_categories(n_concepts)

    all_pcs = []
    for i, (ax, (title, subtitle, data_key)) in enumerate(zip(pc_axes, PC_PANELS)):
        pcs, _ = compute_pca(reps[data_key])
        all_pcs.append(pcs)
        plot_scatter_panel(ax, pcs, super_labels, title, subtitle=subtitle,
                           point_size=12, alpha=0.62)
        if i > 0:
            ax.set_ylabel("")

    # Image insets on scatter panels
    draw_image_insets(pc_axes, all_pcs, concept_names)

    # Super-category legend inside first scatter panel
    cat_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=SUPER_COLORS[name],
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=5.5, label=name)
        for name in SUPER_ORDER
    ]
    leg_cat = pc_axes[0].legend(
        handles=cat_handles, loc="upper left",
        ncol=2, fontsize=7.5, frameon=True,
        handletextpad=0.2, columnspacing=0.5, labelspacing=0.2,
        borderpad=0.3, edgecolor="#dddddd", fancybox=False, framealpha=0.90)
    leg_cat.get_frame().set_linewidth(0.3)

    # ── Uniform axis styling ─────────────────────────────────────────────
    for ax in [ax_coarse, ax_compare, ax_data_eff] + pc_axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(axis="both", which="major", width=0.8, length=4)
        ax.tick_params(axis="both", which="minor", width=0.5, length=2.5)

    # ── Panel labels ─────────────────────────────────────────────────────
    for ax, label, x_off, y_off in zip(
        [ax_schematic, ax_coarse, ax_compare, ax_data_eff, pc_axes[0]],
        ["a", "b", "c", "d", "e"],
        [-0.06, -0.18, -0.06, -0.18, -0.10],
        [1.08,  1.08,  1.08,  1.08,  1.10]):
        ax.text(x_off, y_off, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Save ─────────────────────────────────────────────────────────────
    out = f"{OUTPUT_DIR}/figure4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
