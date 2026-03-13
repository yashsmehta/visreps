"""Figure 2: Coarse Representations Are Fundamentally Different.

Composite figure:
  A — 2×3 grid of class-level RDMs (4,8,16,32,64,1000-way CLIP-PCA)
  B — Cross-model RSA: 1K vs coarse (top), projection vs inter-seed coarse (bottom)

Usage:
    python manuscript/figures/fig2/figure2.py
    python manuscript/figures/fig2/figure2.py --recompute-rdms
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
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import torch

sys.path.insert(0, ".")
from visreps.analysis.rsa import compute_rdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDM_CACHE = os.path.join(SCRIPT_DIR, "class_rdm_data.npz")
RSA_CACHE = os.path.join(SCRIPT_DIR, "cross_model_rsa_data.json")
OUTPUT_DIR = SCRIPT_DIR

# ── Config ───────────────────────────────────────────────────────────────
COARSE_CFG_IDS = [4, 8, 16, 32, 64]
CATEGORY_NAMES = [
    "Animals", "Natural World", "Food & Produce",
    "Structures & Architecture", "Domestic & Apparel",
    "Vehicles & Transport", "Tools & Electronics", "General Objects",
]
CATEGORY_COLORS = [
    "#c0392b", "#27ae60", "#e67e22", "#2980b9",
    "#8e44ad", "#7f8c8d", "#d4527a", "#95a5a6",
]
COLOR_INTERSEED_1K = "#555555"
COLOR_INTERSEED_COARSE = "#66a61e"   # muted green (colorbrewer Dark2)
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
BAR_COLORS = {
    2: "#bdd7e7", 4: "#9ecae1", 8: "#6baed6",
    16: "#4292c6", 32: "#2171b5", 64: "#084594",
}


# ── Rounded bar ──────────────────────────────────────────────────────────

def _draw_rounded_bar(ax, x, height, width, color, hatch="", zorder=3,
                      edgecolor="#333333", alpha=1.0, linewidth=0.6):
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.012, rounding_size=0.05),
        facecolor=color, edgecolor=edgecolor, alpha=alpha,
        linewidth=linewidth, hatch=hatch, mutation_aspect=0.04, zorder=zorder,
    )
    ax.add_patch(rect)


# ── RDM helpers ──────────────────────────────────────────────────────────

def rank_transform(rdm):
    n = rdm.shape[0]
    triu = np.triu_indices(n, k=1)
    ranks = rankdata(rdm[triu]) / rdm[triu].size
    ranked = np.zeros_like(rdm)
    ranked[triu] = ranks
    ranked.T[triu] = ranks
    return ranked


def build_sort_order(categories, rdm):
    unique_cats = sorted(set(categories))
    sorted_indices, block_boundaries = [], []
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
                 width_frac=0.022, gap_frac=0.006):
    w = n * width_frac
    gap = n * gap_frac
    for start, cat, size, _ in block_boundaries:
        color = CATEGORY_COLORS[cat] if cat < len(CATEGORY_COLORS) else "#888888"
        if side == "left":
            ax.add_patch(mpatches.Rectangle(
                (-w - gap, start - 0.5), w, size,
                facecolor=color, edgecolor="none", clip_on=False))
        else:
            ax.add_patch(mpatches.Rectangle(
                (start - 0.5, n - 0.5 + gap), size, w,
                facecolor=color, edgecolor="none", clip_on=False))


def draw_boundaries(ax, block_boundaries, n):
    for start, _, _, _ in block_boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color="white", lw=0.6, alpha=0.75)
            ax.axvline(start - 0.5, color="white", lw=0.6, alpha=0.75)


def plot_rdm_panel(ax, rdm, block_boundaries, n, title):
    rdm_ranked = rank_transform(rdm)
    im = ax.imshow(rdm_ranked, cmap="magma", interpolation="nearest",
                   aspect="equal", rasterized=True, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6, color="#1a1a1a")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_boundaries(ax, block_boundaries, n)
    draw_sidebar(ax, block_boundaries, n, side="left")
    draw_sidebar(ax, block_boundaries, n, side="bottom")
    return im


# ── Panel B — bar plot formatting ────────────────────────────────────────

def _format_bar_axes(ax, valid_cfgs, y_max, ylabel=True, xlabel=True):
    """Shared axis formatting for bar sub-panels."""
    n = len(valid_cfgs)
    x = np.arange(n, dtype=float)
    ax.set_xticks(x)
    if xlabel:
        ax.set_xticklabels([str(c) for c in valid_cfgs], fontsize=8)
        ax.set_xlabel("Number of coarse classes", fontsize=9, labelpad=5)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.55, n - 0.45)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: "" if np.isclose(v, 0) else f"{v:.2f}"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="out", length=3.5,
                   width=0.6, labelsize=7.5)
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.tick_params(axis="x", direction="out", bottom=False, length=0, pad=3)
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=4)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)


def plot_panel_b_rsa(ax):
    """Top sub-panel: cross-model RSA (1000-way vs coarse)."""
    with open(RSA_CACHE) as f:
        results = json.load(f)

    comparisons = results["comparisons"]
    valid_cfgs = [c for c in COARSE_CFGS if str(c) in comparisons
                  and "error" not in comparisons[str(c)]]
    x = np.arange(len(valid_cfgs), dtype=float)
    bar_width = 0.56

    interseed_1k = results.get("interseed_1k", np.nan)
    if not np.isnan(interseed_1k):
        ax.axhline(interseed_1k, color=COLOR_INTERSEED_1K, linestyle="--",
                    linewidth=1.0, zorder=1, alpha=0.55)

    rsa_vals = []
    for cfg_id in valid_cfgs:
        comp = comparisons[str(cfg_id)]
        rsa_vals.append(comp.get("cross_1k_coarse", np.nan))

    for i, (xp, val, cfg_id) in enumerate(zip(x, rsa_vals, valid_cfgs)):
        if np.isnan(val) or val <= 0:
            continue
        _draw_rounded_bar(ax, xp, val, bar_width, BAR_COLORS.get(cfg_id, "#3182bd"))

    ax.set_title("1000-way vs. coarse", fontsize=10, fontweight="bold",
                 pad=7, color="#1a1a1a")
    y_max = interseed_1k * 1.10 if not np.isnan(interseed_1k) else 0.8
    _format_bar_axes(ax, valid_cfgs, y_max, xlabel=False)

    legend_handles = [
        Line2D([], [], color=COLOR_INTERSEED_1K, linestyle="--", linewidth=1.0,
               alpha=0.55, label=f"Inter-seed 1K ({interseed_1k:.2f})"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left",
              frameon=True, edgecolor="#dddddd", fancybox=False,
              handletextpad=0.4, borderpad=0.35, labelspacing=0.3,
              framealpha=0.92)
    ax.get_legend().get_frame().set_linewidth(0.5)


def plot_panel_b_projection(ax):
    """Bottom sub-panel: projected-1K vs coarse AND inter-seed coarse similarity."""
    with open(RSA_CACHE) as f:
        results = json.load(f)

    comparisons = results["comparisons"]
    valid_cfgs = [c for c in COARSE_CFGS if str(c) in comparisons
                  and "error" not in comparisons[str(c)]]
    n = len(valid_cfgs)
    x = np.arange(n, dtype=float)
    bar_width = 0.28
    offset = 0.155

    # Gather data
    proj_vals, interseed_vals, n_pcs_labels = [], [], []
    for cfg_id in valid_cfgs:
        comp = comparisons[str(cfg_id)]
        proj_vals.append(comp.get("projected_1k_coarse", np.nan))
        interseed_vals.append(comp.get("interseed_coarse", np.nan))
        n_pcs_labels.append(comp.get("n_pcs_used", int(np.log2(cfg_id))))

    # Projected-1K vs coarse bars (hatched, left position)
    original_hatch_color = plt.rcParams.get("hatch.color")
    plt.rcParams["hatch.color"] = "#555555"

    for i, (xp, val, cfg_id) in enumerate(zip(x, proj_vals, valid_cfgs)):
        if np.isnan(val) or val <= 0:
            continue
        _draw_rounded_bar(ax, xp - offset, val, bar_width,
                          BAR_COLORS.get(cfg_id, "#3182bd"),
                          hatch="///", edgecolor="#333333")

    if original_hatch_color is not None:
        plt.rcParams["hatch.color"] = original_hatch_color

    # Inter-seed coarse bars (solid, right position)
    for i, (xp, val, cfg_id) in enumerate(zip(x, interseed_vals, valid_cfgs)):
        if np.isnan(val) or val <= 0:
            continue
        _draw_rounded_bar(ax, xp + offset, val, bar_width,
                          COLOR_INTERSEED_COARSE, edgecolor="#333333")

    # k= annotations above projection bars (skip if bar is too short to label cleanly)
    for i, (val, k) in enumerate(zip(proj_vals, n_pcs_labels)):
        if not np.isnan(val) and val > 0.05:
            ax.text(i - offset, val + 0.012, f"k={k}", ha="center", va="bottom",
                    fontsize=5.5, color="#444444", fontstyle="italic")

    ax.set_title("Projection vs. inter-seed coarse", fontsize=10,
                 fontweight="bold", pad=7, color="#1a1a1a")

    y_max = max(v for v in interseed_vals if not np.isnan(v)) * 1.10
    _format_bar_axes(ax, valid_cfgs, y_max)

    # Legend — hatched patch matches the actual projection bars
    proj_patch = mpatches.Patch(facecolor="#4292c6", edgecolor="#333333",
                                linewidth=0.5, hatch="///",
                                label="Projected-1K vs. coarse")
    interseed_patch = mpatches.Patch(facecolor=COLOR_INTERSEED_COARSE,
                                      edgecolor="#333333", linewidth=0.5,
                                      label="Inter-seed coarse")
    ax.legend(handles=[proj_patch, interseed_patch], fontsize=6.5,
              loc="lower left", frameon=True, edgecolor="#dddddd",
              fancybox=False, handletextpad=0.4, borderpad=0.35,
              labelspacing=0.3, framealpha=0.92,
              bbox_to_anchor=(0.0, 0.0))
    ax.get_legend().get_frame().set_linewidth(0.5)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-rdms", action="store_true")
    args = parser.parse_args()

    if args.recompute_rdms or not os.path.exists(RDM_CACHE):
        print("RDM cache not found. Run plot_class_rdms.py first.")
        return
    if not os.path.exists(RSA_CACHE):
        print("RSA cache not found. Run plot_cross_model_rsa.py first.")
        return

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ── Load RDM data ──
    data = np.load(RDM_CACHE)
    centroids_1k = data["centroids_1k"]
    categories = data["categories"]
    valid = categories >= 0
    centroids_1k = centroids_1k[valid]
    categories = categories[valid]
    n_classes = len(categories)

    # Compute 1000-way RDM and sort order
    rdm_1k = compute_rdm(torch.tensor(centroids_1k, dtype=torch.float32)).numpy()
    sort_idx, block_boundaries = build_sort_order(categories, rdm_1k)
    rdm_1k_sorted = rdm_1k[np.ix_(sort_idx, sort_idx)]

    # Load coarse centroids and compute sorted RDMs
    coarse_rdms = {}
    for cfg_id in COARSE_CFG_IDS:
        key = f"centroids_{cfg_id}"
        if key in data:
            cent = data[key][valid]
            rdm = compute_rdm(torch.tensor(cent, dtype=torch.float32)).numpy()
            coarse_rdms[cfg_id] = rdm[np.ix_(sort_idx, sort_idx)]

    # ── Figure layout ──
    fig = plt.figure(figsize=(14, 8))

    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.35, 0.65],
        wspace=0.22,
        left=0.02, right=0.97, top=0.94, bottom=0.02,
    )

    # ── Panel A: 2×3 RDM grid with category legend below ──
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 0],
        height_ratios=[1.0, 0.055],
        hspace=0.06,
    )
    gs_rdms = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_left[0, 0],
        wspace=0.10, hspace=0.16,
    )

    rdm_order = COARSE_CFG_IDS + [1000]  # 4, 8, 16, 32, 64, 1000
    rdm_titles = [f"{c}-way" for c in COARSE_CFG_IDS] + ["1000-way"]
    rdm_axes = []
    im = None

    for i, (cfg_id, title) in enumerate(zip(rdm_order, rdm_titles)):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs_rdms[row, col])
        rdm_axes.append(ax)
        rdm = rdm_1k_sorted if cfg_id == 1000 else coarse_rdms.get(cfg_id)
        if rdm is not None:
            im = plot_rdm_panel(ax, rdm, block_boundaries, n_classes, title)
        else:
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6,
                         color="#1a1a1a")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999")
            ax.set_xticks([]); ax.set_yticks([])

    # Colorbar — inset next to last RDM in top row
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_last_top = rdm_axes[2]  # top-right RDM
    ax_cb = inset_axes(ax_last_top, width="3.5%", height="100%",
                       loc="center right",
                       bbox_to_anchor=(0.09, 0, 1, 1),
                       bbox_transform=ax_last_top.transAxes, borderpad=0)
    cb = plt.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=6.5, length=2.5, width=0.5, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
    cb.ax.set_ylabel("Dissimilarity (rank)", fontsize=7, labelpad=6,
                      rotation=270, va="bottom")

    # Category legend (compact, below RDM grid)
    ax_legend = fig.add_subplot(gs_left[1, 0])
    ax_legend.axis("off")
    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="none", label=name)
        for i, name in enumerate(CATEGORY_NAMES)
    ]
    ax_legend.legend(handles=legend_handles, loc="center", fontsize=7,
                     frameon=False, ncol=4, columnspacing=0.9,
                     handlelength=1.2, handleheight=0.9, labelspacing=0.35,
                     bbox_to_anchor=(0.48, 0.5))

    # ── Panel B: two stacked bar plots ──
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[1.0, 1.0],
        hspace=0.38,
    )
    ax_rsa = fig.add_subplot(gs_right[0, 0])
    ax_proj = fig.add_subplot(gs_right[1, 0])
    plot_panel_b_rsa(ax_rsa)
    plot_panel_b_projection(ax_proj)

    # ── Panel labels ──
    rdm_axes[0].text(-0.06, 1.10, "A", transform=rdm_axes[0].transAxes,
                     fontsize=15, fontweight="bold", va="top", ha="left",
                     fontfamily="sans-serif")
    ax_rsa.text(-0.12, 1.12, "B", transform=ax_rsa.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif")

    # ── Save ──
    out = os.path.join(OUTPUT_DIR, "figure2.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
