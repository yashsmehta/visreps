"""Figure 2: Coarse Representations Are Fundamentally Different.

Composite figure with two panel groups:
  A — Class-level RDM comparison (1000-way vs. 8-way CLIP-PCA)
  B — Cross-model RSA bars (1K vs coarse, projected-1K vs coarse)

Layout (single row):
  Left:  [A1: 1000-way RDM] [A2: 8-way RDM] [colorbar] + category legend below
  Right: [B: projected-1K vs coarse RSA bars]

Usage:
    python manuscript/figures/fig2/figure2.py
    python manuscript/figures/fig2/figure2.py --recompute-rdms   # recompute RDM data
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import torch

sys.path.insert(0, ".")
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDM_CACHE = os.path.join(SCRIPT_DIR, "class_rdm_data.npz")
RSA_CACHE = os.path.join(SCRIPT_DIR, "cross_model_rsa_data.json")
OUTPUT_DIR = SCRIPT_DIR

# ── Shared config ────────────────────────────────────────────────────────
COARSE_CFG_ID = 8

CATEGORY_NAMES = [
    "Animals", "Natural World", "Food & Produce",
    "Structures & Architecture", "Domestic & Apparel",
    "Vehicles & Transport", "Tools & Electronics", "General Objects",
]
CATEGORY_COLORS = [
    "#D84315", "#2E7D32", "#F57C00", "#1565C0",
    "#7B1FA2", "#8D6E63", "#E91E90", "#78909C",
]

COLOR_INTERSEED = "#9E9E9E"
COLOR_CROSS = "#1565C0"
COLOR_PROJECTED = "#E8890C"

COARSE_CFGS = [2, 4, 8, 16, 32, 64]


# ── Rounded bar helper ───────────────────────────────────────────────────

def draw_rounded_bars(ax, x_positions, heights, width, color, label,
                      edgecolor="none", linewidth=0.5, radius=0.06, zorder=3):
    """Draw bars with rounded top corners using FancyBboxPatch."""
    for i, (xp, h) in enumerate(zip(x_positions, heights)):
        if np.isnan(h) or h <= 0:
            continue
        box = FancyBboxPatch(
            (xp - width / 2, 0), width, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
            zorder=zorder,
            label=label if i == 0 else None,
        )
        ax.add_patch(box)


# ── Panel A helpers (class-level RDMs) ───────────────────────────────────

def rank_transform(rdm):
    """Rank upper triangle, mirror to lower, scale to [0, 1]."""
    n = rdm.shape[0]
    triu = np.triu_indices(n, k=1)
    ranks = rankdata(rdm[triu]) / rdm[triu].size
    ranked = np.zeros_like(rdm)
    ranked[triu] = ranks
    ranked.T[triu] = ranks
    return ranked


def build_sort_order(categories, rdm):
    """Sort classes by category, then hierarchical clustering within each."""
    unique_cats = sorted(set(categories))
    sorted_indices = []
    block_boundaries = []
    offset = 0
    for cat in unique_cats:
        member_idx = np.where(categories == cat)[0]
        if len(member_idx) <= 2:
            order = member_idx
        else:
            sub_rdm = rdm[np.ix_(member_idx, member_idx)]
            sub_condensed = squareform(sub_rdm, checks=False)
            sub_condensed = np.maximum(sub_condensed, 0)
            sub_order = leaves_list(linkage(sub_condensed, method="average"))
            order = member_idx[sub_order]
        cat_name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else f"Cat {cat}"
        block_boundaries.append((offset, cat, len(order), cat_name))
        sorted_indices.extend(order)
        offset += len(order)
    return np.array(sorted_indices), block_boundaries


def draw_sidebar(ax, block_boundaries, n, side="left",
                 width_frac=0.02, gap_frac=0.006):
    """Draw colored sidebar along RDM edge."""
    w = n * width_frac
    gap = n * gap_frac
    for start, cat, size, _ in block_boundaries:
        color = CATEGORY_COLORS[cat] if cat < len(CATEGORY_COLORS) else "#888888"
        if side == "left":
            rect = mpatches.Rectangle(
                (-w - gap, start - 0.5), w, size,
                facecolor=color, edgecolor="none", clip_on=False)
        else:
            rect = mpatches.Rectangle(
                (start - 0.5, n - 0.5 + gap), size, w,
                facecolor=color, edgecolor="none", clip_on=False)
        ax.add_patch(rect)


def draw_boundaries(ax, block_boundaries, n, color="white", lw=0.4, alpha=0.6):
    """Draw thin lines at category boundaries."""
    for start, _, size, _ in block_boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color=color, lw=lw, alpha=alpha)
            ax.axvline(start - 0.5, color=color, lw=lw, alpha=alpha)


def plot_rdm_panel(ax, rdm_ranked, block_boundaries, n, title):
    """Draw a single RDM panel with category annotations."""
    im = ax.imshow(rdm_ranked, cmap="magma", interpolation="nearest",
                   aspect="equal", rasterized=True, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_boundaries(ax, block_boundaries, n)
    draw_sidebar(ax, block_boundaries, n, side="left")
    draw_sidebar(ax, block_boundaries, n, side="bottom")
    return im


def plot_panel_a(axes_rdm, ax_cb, ax_legend):
    """Plot Panel A: class-level RDM comparison."""
    data = np.load(RDM_CACHE)
    centroids_1k = data["centroids_1k"]
    centroids_coarse = data["centroids_coarse"]
    categories = data["categories"]

    valid = categories >= 0
    centroids_1k = centroids_1k[valid]
    centroids_coarse = centroids_coarse[valid]
    categories = categories[valid]
    n_classes = len(categories)

    # Compute RDMs
    rdm_1k = compute_rdm(torch.tensor(centroids_1k, dtype=torch.float32)).numpy()
    rdm_coarse = compute_rdm(torch.tensor(centroids_coarse, dtype=torch.float32)).numpy()

    # Cross-model RSA
    rsa = compute_rdm_correlation(
        torch.tensor(rdm_1k), torch.tensor(rdm_coarse), correlation="Spearman")

    # Sort by category
    sort_idx, block_boundaries = build_sort_order(categories, rdm_1k)
    rdm_1k_sorted = rdm_1k[np.ix_(sort_idx, sort_idx)]
    rdm_coarse_sorted = rdm_coarse[np.ix_(sort_idx, sort_idx)]

    rdm_1k_ranked = rank_transform(rdm_1k_sorted)
    rdm_coarse_ranked = rank_transform(rdm_coarse_sorted)

    # Plot RDMs
    im1 = plot_rdm_panel(axes_rdm[0], rdm_1k_ranked, block_boundaries,
                          n_classes, "1000-way Model")
    plot_rdm_panel(axes_rdm[1], rdm_coarse_ranked, block_boundaries,
                   n_classes, f"{COARSE_CFG_ID}-way CLIP-PCA Model")

    # Colorbar
    cb = plt.colorbar(im1, cax=ax_cb)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
    cb.set_label("Dissimilarity (rank)", fontsize=7, labelpad=3)

    # Category legend
    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="none",
                       label=name)
        for i, name in enumerate(CATEGORY_NAMES)
    ]
    ax_legend.legend(handles=legend_handles, loc="center", fontsize=7,
                     frameon=False, ncol=4, columnspacing=1.2,
                     handlelength=1.1, handleheight=0.8,
                     title="WordNet Super-Categories",
                     title_fontproperties={"size": 8, "weight": "bold"})

    return rsa


# ── Panel B helpers (cross-model RSA bars) ───────────────────────────────

def plot_panel_b(ax):
    """Plot Panel B: projected-1K vs coarse RSA bars.

    Shows how well a low-rank projection of the 1K model (onto k = log2(n_classes)
    PCs) matches the coarse model. Inter-seed 1K baseline as dashed reference.
    """
    with open(RSA_CACHE) as f:
        results = json.load(f)

    comparisons = results["comparisons"]
    method = results.get("method", "spearman")

    valid_cfgs = [c for c in COARSE_CFGS if str(c) in comparisons
                  and "error" not in comparisons[str(c)]]
    n = len(valid_cfgs)
    x = np.arange(n, dtype=float)
    bar_width = 0.45

    # Inter-seed baseline
    interseed_1k = results.get("interseed_1k", np.nan)
    if not np.isnan(interseed_1k):
        ax.axhline(interseed_1k, color=COLOR_INTERSEED, linestyle="--",
                    linewidth=1.2, zorder=1,
                    label=f"Inter-seed 1K baseline ({interseed_1k:.2f})")

    # Single bars: projected-1K vs coarse only
    proj_vals, n_pcs_labels = [], []
    for cfg_id in valid_cfgs:
        comp = comparisons[str(cfg_id)]
        proj_vals.append(comp.get("projected_1k_coarse", np.nan))
        n_pcs_labels.append(comp.get("n_pcs_used", int(np.log2(cfg_id))))

    draw_rounded_bars(ax, x, proj_vals, bar_width,
                      COLOR_PROJECTED, "Projected-1K vs. Coarse", radius=0.05)

    # Annotate number of PCs above each bar
    for i, (val, k) in enumerate(zip(proj_vals, n_pcs_labels)):
        if not np.isnan(val):
            ax.text(i, val + 0.012, f"k={k}", ha="center", va="bottom",
                    fontsize=7, color="#555555", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in valid_cfgs], fontsize=8)
    ax.set_xlabel("Number of Classes", fontsize=8.5, labelpad=5)
    method_label = method.capitalize()
    ax.set_ylabel(f"RSA ({method_label} " + r"$\rho$)", fontsize=8.5, labelpad=5)
    ax.set_title("Cross-Model RSA (FC1)", fontsize=9, fontweight="semibold", pad=6)

    ax.set_ylim(0, max(interseed_1k + 0.1, 0.9) if not np.isnan(interseed_1k) else 0.5)
    ax.set_xlim(-0.6, n - 0.4)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", labelsize=8, length=4, width=0.8)
    ax.tick_params(axis="y", which="minor", length=2.5, width=0.5)
    ax.yaxis.grid(True, which="major", color="#E8E8E8", linewidth=0.5, zorder=0)
    ax.yaxis.grid(True, which="minor", color="#F2F2F2", linewidth=0.3, zorder=0)

    ax.legend(loc="upper left", fontsize=6.5, frameon=True,
              edgecolor="#DDDDDD", fancybox=False, handletextpad=0.4,
              framealpha=0.95, borderpad=0.5, labelspacing=0.35)
    sns.despine(ax=ax, offset=5)


# ── Main figure assembly ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-rdms", action="store_true",
                        help="Recompute class RDM data (run plot_class_rdms.py first)")
    args = parser.parse_args()

    if args.recompute_rdms or not os.path.exists(RDM_CACHE):
        print("RDM cache not found. Run plot_class_rdms.py first:")
        print("  python manuscript/figures/fig2/plot_class_rdms.py")
        return

    if not os.path.exists(RSA_CACHE):
        print("RSA cache not found. Run plot_cross_model_rsa.py first:")
        print("  python manuscript/figures/fig2/plot_cross_model_rsa.py")
        return

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    # Layout: single row — [RDMs + legend | bar chart]
    fig = plt.figure(figsize=(12.5, 5.5))

    # Outer grid: left (RDMs) and right (bar chart)
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.3, 0.7],
        wspace=0.22,
    )

    # Left: 2 rows (RDMs + category legend), 3 cols (RDM1, RDM2, colorbar)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_outer[0, 0],
        width_ratios=[1, 1, 0.035],
        height_ratios=[1.0, 0.12],
        hspace=0.10, wspace=0.14,
    )

    ax_rdm1 = fig.add_subplot(gs_left[0, 0])
    ax_rdm2 = fig.add_subplot(gs_left[0, 1])
    ax_cb = fig.add_subplot(gs_left[0, 2])
    ax_legend = fig.add_subplot(gs_left[1, :])
    ax_legend.axis("off")

    rsa_score = plot_panel_a([ax_rdm1, ax_rdm2], ax_cb, ax_legend)

    # RSA annotation in the gap between the two RDMs
    pos1 = ax_rdm1.get_position()
    pos2 = ax_rdm2.get_position()
    gap_x = (pos1.x1 + pos2.x0) / 2
    gap_y = (pos1.y0 + pos1.y1) / 2
    fig.text(gap_x, gap_y,
             f"$\\rho_s$ = {rsa_score:.3f}",
             ha="center", va="center", fontsize=8, color="#333333",
             fontweight="semibold", rotation=90,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB",
                       alpha=0.95, linewidth=0.6))

    # Right: bar chart aligned with RDMs (same 2-row split)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[1.0, 0.12],
        hspace=0.10,
    )
    ax_bars = fig.add_subplot(gs_right[0, 0])
    plot_panel_b(ax_bars)
    # Hide bottom-right to match legend row
    ax_blank = fig.add_subplot(gs_right[1, 0])
    ax_blank.set_visible(False)

    # Panel labels
    ax_rdm1.text(-0.10, 1.06, "A", transform=ax_rdm1.transAxes,
                 fontsize=14, fontweight="bold", va="top")
    ax_bars.text(-0.16, 1.06, "B", transform=ax_bars.transAxes,
                 fontsize=14, fontweight="bold", va="top")

    # Save
    out = os.path.join(OUTPUT_DIR, "figure2.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
