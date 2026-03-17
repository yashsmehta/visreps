"""Figure 5: Per-Concept Alignment Analysis.

Layout (2 rows):
  Row 1:  [Behavioral RDM]  [CLIP 8-class RDM]  [1000-class RDM]  [colorbar]
          [              super-category legend row                  ]
  Row 2:  [Scatter plot]                          [Histogram]

Panel A: Category-sorted RDMs — Behavioral vs CLIP 8-class vs 1000-class
         (concepts grouped by 8 semantic super-categories derived from THINGS-27)
Panel B: Per-concept scatter — CLIP 8-class vs 1000-way per-concept RSA
Panel C: Histogram of per-concept advantage (delta rho)

Usage:
    python manuscript/figures/fig5/figure5.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style
from things_utils import compute_things_data, plot_scatter_panel
from experiments.things_visualizations.plot_rdms_categorized import (
    load_categories, draw_category_sidebar, draw_boundary_lines,
    rank_transform,
)

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig5"

# Super-category groupings (ordered for display)
SUPER_CAT_ORDER = [
    "Living things",
    "Body & apparel",
    "Food & drink",
    "Home",
    "Tools & equipment",
    "Vehicles",
    "Tech & leisure",
    "Other",
]

FINE_TO_SUPER = {
    "animal": "Living things", "plant": "Living things",
    "body part": "Body & apparel", "clothing": "Body & apparel",
    "clothing accessory": "Body & apparel",
    "food": "Food & drink", "dessert": "Food & drink",
    "drink": "Food & drink", "kitchen appliance": "Food & drink",
    "kitchen tool": "Food & drink",
    "furniture": "Home", "home decor": "Home", "container": "Home",
    "tool": "Tools & equipment", "sports equipment": "Tools & equipment",
    "medical equipment": "Tools & equipment",
    "office supply": "Tools & equipment", "weapon": "Tools & equipment",
    "vehicle": "Vehicles", "part of car": "Vehicles",
    "electronic device": "Tech & leisure",
    "musical instrument": "Tech & leisure", "toy": "Tech & leisure",
    "Other": "Other",
}

# Distinguishable palette for 8 super-categories
SUPER_PALETTE = {
    "Living things":      "#2ca02c",  # green
    "Body & apparel":     "#9467bd",  # purple
    "Food & drink":       "#d62728",  # red
    "Home":               "#ff7f0e",  # orange
    "Tools & equipment":  "#1f77b4",  # blue
    "Vehicles":           "#8c564b",  # brown
    "Tech & leisure":     "#17becf",  # cyan
    "Other":              "#bdbdbd",  # grey
}


def _build_super_sort_order(fine_categories, behav_rdm):
    """Sort concepts by super-category, then hierarchical clustering within."""
    super_cats = np.array([FINE_TO_SUPER.get(c, "Other") for c in fine_categories])

    sorted_indices = []
    block_boundaries = []
    offset = 0

    for scat in SUPER_CAT_ORDER:
        member_idx = np.where(super_cats == scat)[0]
        if len(member_idx) == 0:
            continue
        if len(member_idx) <= 2:
            order = member_idx
        else:
            sub_rdm = behav_rdm[np.ix_(member_idx, member_idx)]
            sub_condensed = squareform(sub_rdm, checks=False)
            sub_order = leaves_list(linkage(sub_condensed, method="average"))
            order = member_idx[sub_order]

        block_boundaries.append((offset, scat, len(order)))
        sorted_indices.extend(order)
        offset += len(order)

    return np.array(sorted_indices), block_boundaries


def _draw_rdm(ax, rdm, title, subtitle, block_boundaries, n, super_cats_used,
              subtitle_italic=False):
    """Draw a single RDM panel with super-category sidebars."""
    im = ax.imshow(rdm, cmap="magma", interpolation="nearest", aspect="equal",
                   rasterized=True, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=18,
                 fontfamily="sans-serif")
    if subtitle:
        style = "italic" if subtitle_italic else "normal"
        ax.text(0.5, 1.013, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#555555",
                fontstyle=style)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cat_colors = [SUPER_PALETTE[c] for c in super_cats_used]
    cat_to_idx = {c: i for i, c in enumerate(super_cats_used)}

    draw_boundary_lines(ax, block_boundaries, n, color="white",
                        lw=0.45, alpha=0.80)
    draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                          side="left", width_frac=0.028, gap_frac=0.005)
    draw_category_sidebar(ax, block_boundaries, n, cat_colors, cat_to_idx,
                          side="bottom", width_frac=0.028, gap_frac=0.005)
    return im


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    print("Computing THINGS data for per-concept analysis...")
    precomputed = compute_things_data()

    # ── Build super-category-sorted RDMs ─────────────────────────────
    fine_sort_idx = precomputed["sort_idx"]
    unsort = np.argsort(fine_sort_idx)
    rdms_ranked = precomputed["rdms_ranked"]

    rdm_behav_orig = rdms_ranked["Behavioral"][np.ix_(unsort, unsort)]
    rdm_clip8_orig = rdms_ranked["CLIP 8-class"][np.ix_(unsort, unsort)]
    rdm_1k_orig = rdms_ranked["1000-class"][np.ix_(unsort, unsort)]

    fine_categories = load_categories()
    super_sort_idx, super_boundaries = _build_super_sort_order(
        fine_categories, rdm_behav_orig
    )

    super_cats_used = [scat for _, scat, _ in super_boundaries]

    rdm_behav_super = rdm_behav_orig[np.ix_(super_sort_idx, super_sort_idx)]
    rdm_clip8_super = rdm_clip8_orig[np.ix_(super_sort_idx, super_sort_idx)]
    rdm_1k_super = rdm_1k_orig[np.ix_(super_sort_idx, super_sort_idx)]
    n = rdm_behav_super.shape[0]

    rsa_scores = precomputed["rsa_scores"]

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 13.2))
    fig.patch.set_facecolor("white")

    gs_outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 0.04, 1.10],
        hspace=0.08,
        left=0.05, right=0.96, top=0.96, bottom=0.05,
    )

    # ── Row 1: Three RDMs + colorbar ─────────────────────────────────
    gs_rdm = gridspec.GridSpecFromSubplotSpec(
        3, 4, subplot_spec=gs_outer[0],
        width_ratios=[1, 1, 1, 0.035],
        height_ratios=[0.08, 0.84, 0.08],
        wspace=0.12, hspace=0,
    )
    ax_rdm_behav = fig.add_subplot(gs_rdm[0:3, 0])
    ax_rdm_clip8 = fig.add_subplot(gs_rdm[0:3, 1])
    ax_rdm_1k = fig.add_subplot(gs_rdm[0:3, 2])
    ax_cb = fig.add_subplot(gs_rdm[1, 3])

    im = _draw_rdm(ax_rdm_behav, rdm_behav_super, "Behavioral",
                    "(ground truth)", super_boundaries, n, super_cats_used,
                    subtitle_italic=True)
    _draw_rdm(ax_rdm_clip8, rdm_clip8_super, "CLIP 8-class",
              f"$\\rho_s$ = {rsa_scores['CLIP 8-class']:.3f}",
              super_boundaries, n, super_cats_used)
    _draw_rdm(ax_rdm_1k, rdm_1k_super, "1000-class",
              f"$\\rho_s$ = {rsa_scores['1000-class']:.3f}",
              super_boundaries, n, super_cats_used)

    # Shared colorbar
    cb = plt.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=8, length=3, width=0.5, pad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
    cb.set_label("Dissimilarity (rank)", fontsize=8.5, labelpad=8)

    # ── Category legend row ──────────────────────────────────────────
    ax_legend = fig.add_subplot(gs_outer[1])
    ax_legend.axis("off")

    legend_handles = []
    for scat in super_cats_used:
        legend_handles.append(mpatches.Patch(
            facecolor=SUPER_PALETTE[scat], edgecolor="#aaaaaa",
            linewidth=0.5, label=scat))

    leg_cat = ax_legend.legend(
        handles=legend_handles, loc="center", fontsize=8,
        frameon=False, ncol=len(super_cats_used), columnspacing=1.0,
        handlelength=1.1, handleheight=0.75, labelspacing=0.35,
        handletextpad=0.4,
    )
    leg_cat._legend_box.sep = 3

    # ── Row 2: Scatter + Histogram ───────────────────────────────────
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[2],
        width_ratios=[1.1, 1.0],
        wspace=0.28,
    )
    ax_scatter = fig.add_subplot(gs_bottom[0, 0])
    ax_hist = fig.add_subplot(gs_bottom[0, 1])

    plot_scatter_panel(ax_scatter, ax_hist, precomputed)

    # Override scatter title padding for this layout
    ax_scatter.set_title("Per-Concept Alignment", fontsize=12.5,
                         fontweight="semibold", pad=14)

    # Override histogram formatting for standalone display
    ax_hist.set_xlabel(
        r"$\Delta\rho_s$ (CLIP 8-class $-$ 1000-class)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.tick_params(axis="both", labelsize=9.5, length=4, width=0.8)
    ax_hist.set_title("Per-Concept Advantage", fontsize=12.5,
                      fontweight="semibold", pad=14)
    sns.despine(ax=ax_hist, offset=5)

    # ── Panel labels ─────────────────────────────────────────────────
    ax_rdm_behav.text(-0.07, 1.12, "A", transform=ax_rdm_behav.transAxes,
                      fontsize=18, fontweight="bold", va="top", ha="left",
                      family="sans-serif")
    ax_scatter.text(-0.11, 1.07, "B", transform=ax_scatter.transAxes,
                    fontsize=18, fontweight="bold", va="top", ha="left",
                    family="sans-serif")
    ax_hist.text(-0.11, 1.07, "C", transform=ax_hist.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="left",
                 family="sans-serif")

    out = f"{OUTPUT_DIR}/figure5.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
