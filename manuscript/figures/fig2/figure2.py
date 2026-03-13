"""Figure 2: Coarse Representations Are Fundamentally Different.

Composite figure:
  A — 2×3 grid of class-level RDMs (4,8,16,32,64,1000-way CLIP-PCA)
  B — Cross-model RSA: 1K vs coarse (top), projection vs inter-seed coarse (bottom)
  C — PC1/PC2 scatter: fine-grained (top) vs coarse (bottom) with image insets

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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import torch

sys.path.insert(0, ".")
from visreps.analysis.rsa import compute_rdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
RDM_CACHE = os.path.join(SCRIPT_DIR, "class_rdm_data.npz")
RSA_CACHE = os.path.join(SCRIPT_DIR, "cross_model_rsa_data.json")
PC_DATA = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                       "2pcs_compare", "data_4way_alexnet.npz")
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


# ── Panel C — PC scatter with image insets ────────────────────────────────

PC_COLORS = ["#00A896", "#7B68EE", "#E8963E", "#D64045"]
N_INSETS = 3      # images per class
INSET_LAYER = "fc1"


def align_pc_projections(pretrained_pcs, pretrained_var, trained_pcs, labels,
                         n_classes):
    """Align the pretrained PCA projection to best match the trained one
    using Procrustes rotation on class centroids.

    This finds the optimal 2D rotation matrix R that minimizes the distance
    between class centroids: ||pre_centroids @ R - tra_centroids||.
    Unlike axis flips or 90° swaps, Procrustes preserves the PC axis identity
    (PC1 stays PC1) while rotating the point cloud for visual alignment.

    Returns (aligned_pretrained_pcs, pretrained_var, trained_pcs).
    Variance array is unchanged since rotation doesn't swap axes.
    """
    # Compute class centroids in each projection
    pre_cent = np.array([pretrained_pcs[labels == c].mean(axis=0)
                         for c in range(n_classes)])
    tra_cent = np.array([trained_pcs[labels == c].mean(axis=0)
                         for c in range(n_classes)])

    # Center both centroid sets
    pre_mean = pre_cent.mean(axis=0)
    tra_mean = tra_cent.mean(axis=0)
    pre_c = pre_cent - pre_mean
    tra_c = tra_cent - tra_mean

    # Optimal rotation via SVD: R = V @ U^T where U S V^T = SVD(tra^T @ pre)
    U, _, Vt = np.linalg.svd(tra_c.T @ pre_c)
    R = (U @ Vt).T  # 2x2 rotation matrix

    # Ensure proper rotation (det = +1), not reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = (U @ Vt).T

    # Apply rotation to all pretrained points
    aligned_pcs = pretrained_pcs @ R

    # Verify alignment
    aligned_cent = np.array([aligned_pcs[labels == c].mean(axis=0)
                             for c in range(n_classes)])
    for c in range(n_classes):
        tra_dir = tra_cent[c] - tra_cent.mean(axis=0)
        ali_dir = aligned_cent[c] - aligned_cent.mean(axis=0)
        dot = np.dot(tra_dir, ali_dir)
        if dot < 0:
            print(f"  Warning: Class {c} centroid not well aligned (dot={dot:.3f})")

    return aligned_pcs, pretrained_var, trained_pcs


def _select_inset_indices(pcs, labels, n_classes, n_per_class=N_INSETS):
    """Pick representative points spread across each class cluster.

    Strategy: exclude the outermost 15% of points (boundary zone),
    then from the inner region pick points in different angular sectors
    from the centroid for good spatial coverage.
    """
    indices = []
    for c in range(n_classes):
        mask = np.where(labels == c)[0]
        if len(mask) < n_per_class:
            indices.extend(mask.tolist())
            continue
        class_pcs = pcs[mask]
        centroid = class_pcs.mean(axis=0)
        dists = np.linalg.norm(class_pcs - centroid, axis=1)

        # Keep inner 85% — safe from class boundaries
        dist_threshold = np.percentile(dists, 85)
        inner_idx = np.where(dists < dist_threshold)[0]
        if len(inner_idx) < n_per_class:
            inner_idx = np.argsort(dists)[:max(n_per_class, len(mask) // 2)]

        # Divide into angular sectors for spatial spread
        offsets = class_pcs[inner_idx] - centroid
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        sector_edges = np.linspace(-np.pi, np.pi, n_per_class + 1)
        picks = []
        for s in range(n_per_class):
            in_sector = (angles >= sector_edges[s]) & (angles < sector_edges[s + 1])
            sector_pts = np.where(in_sector)[0]
            if len(sector_pts) == 0:
                continue
            # Pick at ~40th percentile distance (well inside, not at center)
            sector_dists = dists[inner_idx[sector_pts]]
            target = max(0, int(len(sector_dists) * 0.4))
            pick = sector_pts[np.argsort(sector_dists)[target]]
            picks.append(inner_idx[pick])

        # Fallback
        if len(picks) < n_per_class:
            sorted_by_dist = np.argsort(dists[inner_idx])
            step = max(1, len(sorted_by_dist) // n_per_class)
            for i in range(0, len(sorted_by_dist), step):
                if inner_idx[sorted_by_dist[i]] not in picks:
                    picks.append(inner_idx[sorted_by_dist[i]])
                if len(picks) == n_per_class:
                    break

        indices.extend(mask[picks[:n_per_class]].tolist())
    return indices


def _load_thumbnail(path, size=48):
    """Load and resize an image to a square thumbnail."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


# Cache loaded thumbnails to avoid reloading the same images for both panels
_thumb_cache = {}


def _get_thumbnail(path, size=48):
    if path not in _thumb_cache:
        _thumb_cache[path] = _load_thumbnail(path, size)
    return _thumb_cache[path]


def plot_pc_scatter_panel(ax, pcs, var_explained, labels, n_classes, colors,
                          title, img_paths=None, inset_indices=None,
                          point_size=0.5, alpha=0.20, inset_zoom=0.35,
                          fontscale=1.0):
    """Draw one PC1 vs PC2 scatter panel with optional image insets.

    Args:
        inset_indices: list of indices into pcs/labels/img_paths to show as
            image insets. If None, no insets are drawn.
    """
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))
    pcs_s, labels_s = pcs[order], labels[order]

    for c in range(n_classes):
        mask = labels_s == c
        ax.scatter(
            pcs_s[mask, 0], pcs_s[mask, 1],
            c=colors[c], s=point_size, alpha=alpha, edgecolors="none",
            rasterized=True, zorder=2,
        )

    fs = fontscale
    ax.set_xlabel("PC 1", fontsize=7 * fs, labelpad=2)
    ax.set_ylabel("PC 2", fontsize=7 * fs, labelpad=2)
    ax.set_title(title, fontsize=9 * fs, fontweight="bold", pad=5,
                 color="#1a1a1a")
    ax.tick_params(axis="both", labelsize=6 * fs, length=2, width=0.4, pad=1.5)

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.08
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    # Image insets
    if inset_indices is not None and img_paths is not None:
        for ii in inset_indices:
            thumb = _get_thumbnail(img_paths[ii])
            if thumb is None:
                continue
            im_box = OffsetImage(thumb, zoom=inset_zoom)
            im_box.image.axes = ax
            c = colors[int(labels[ii])]
            ab = AnnotationBbox(
                im_box, (pcs[ii, 0], pcs[ii, 1]),
                frameon=True, pad=0.15,
                bboxprops=dict(edgecolor=c, linewidth=1.5,
                               facecolor="white"),
                zorder=6,
            )
            ax.add_artist(ab)


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

    # ── Load PC scatter data ──
    has_pc_data = os.path.exists(PC_DATA)
    if has_pc_data:
        pc_data = np.load(PC_DATA, allow_pickle=True)
        pc_n_classes = int(pc_data["n_classes"])
        pca_labels = pc_data["pca_labels"]
        img_paths = pc_data["img_paths"]
        pretrained_pcs = pc_data[f"{INSET_LAYER}_pretrained_pcs"].copy()
        pretrained_var = pc_data[f"{INSET_LAYER}_pretrained_var"]
        trained_pcs = pc_data[f"{INSET_LAYER}_trained_pcs"].copy()
        trained_var = pc_data[f"{INSET_LAYER}_trained_var"]
        # Align projections: rotate pretrained 90° CCW + flip signs to match
        pretrained_pcs, pretrained_var, trained_pcs = align_pc_projections(
            pretrained_pcs, pretrained_var, trained_pcs,
            pca_labels, pc_n_classes)
        # Select inset images from coarse panel (clear clusters), show in both
        inset_idx = _select_inset_indices(trained_pcs, pca_labels, pc_n_classes)
    else:
        print("Warning: PC scatter data not found, Panel C will be skipped.")

    # ── Figure layout ──
    fig = plt.figure(figsize=(17, 8))

    gs_outer = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.2, 0.55, 0.50],
        wspace=0.20,
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
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[1.0, 1.0],
        hspace=0.38,
    )
    ax_rsa = fig.add_subplot(gs_mid[0, 0])
    ax_proj = fig.add_subplot(gs_mid[1, 0])
    plot_panel_b_rsa(ax_rsa)
    plot_panel_b_projection(ax_proj)

    # ── Panel C: PC scatter with image insets ──
    if has_pc_data:
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_outer[0, 2],
            height_ratios=[1.0, 1.0],
            hspace=0.38,
        )
        ax_pc_pre = fig.add_subplot(gs_right[0, 0])
        ax_pc_coarse = fig.add_subplot(gs_right[1, 0])

        # Same images shown in both panels so viewer sees reorganization
        plot_pc_scatter_panel(
            ax_pc_pre, pretrained_pcs, pretrained_var,
            pca_labels, pc_n_classes, PC_COLORS,
            title="Fine-grained (1000-way)",
            img_paths=img_paths, inset_indices=inset_idx,
        )
        plot_pc_scatter_panel(
            ax_pc_coarse, trained_pcs, trained_var,
            pca_labels, pc_n_classes, PC_COLORS,
            title=f"Coarsened ({pc_n_classes}-way)",
            img_paths=img_paths, inset_indices=inset_idx,
        )

        # Shared legend for scatter panels
        scatter_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PC_COLORS[c],
                   markeredgecolor="none", markersize=5,
                   label=f"Class {c}")
            for c in range(pc_n_classes)
        ]
        ax_pc_coarse.legend(
            handles=scatter_handles, loc="lower center",
            bbox_to_anchor=(0.5, -0.18), ncol=pc_n_classes,
            fontsize=6.5, frameon=False, handletextpad=0.15,
            columnspacing=0.6,
        )

    # ── Panel labels ──
    rdm_axes[0].text(-0.06, 1.10, "A", transform=rdm_axes[0].transAxes,
                     fontsize=15, fontweight="bold", va="top", ha="left",
                     fontfamily="sans-serif")
    ax_rsa.text(-0.12, 1.12, "B", transform=ax_rsa.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif")
    if has_pc_data:
        ax_pc_pre.text(-0.14, 1.12, "C", transform=ax_pc_pre.transAxes,
                       fontsize=15, fontweight="bold", va="top", ha="left",
                       fontfamily="sans-serif")

    # ── Save ──
    out = os.path.join(OUTPUT_DIR, "figure2.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()

    # ── Standalone PC scatter ──
    if has_pc_data:
        save_standalone_pc_scatter(
            pretrained_pcs, pretrained_var, trained_pcs, trained_var,
            pca_labels, pc_n_classes, img_paths, inset_idx,
        )
        save_dense_pc_scatter(
            pretrained_pcs, pretrained_var, trained_pcs, trained_var,
            pca_labels, pc_n_classes, img_paths, n_images=1000,
        )


def save_standalone_pc_scatter(pretrained_pcs, pretrained_var, trained_pcs,
                               trained_var, pca_labels, pc_n_classes,
                               img_paths, inset_idx):
    """Save a standalone version of the PC scatter panels."""
    _thumb_cache.clear()  # fresh cache for standalone sizing
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Same inset images in both panels
    plot_pc_scatter_panel(
        axes[0], pretrained_pcs, pretrained_var,
        pca_labels, pc_n_classes, PC_COLORS,
        title="Fine-grained (1000-way)",
        img_paths=img_paths, inset_indices=inset_idx,
        point_size=0.7, alpha=0.25, inset_zoom=0.40, fontscale=1.15,
    )
    plot_pc_scatter_panel(
        axes[1], trained_pcs, trained_var,
        pca_labels, pc_n_classes, PC_COLORS,
        title=f"Coarsened ({pc_n_classes}-way)",
        img_paths=img_paths, inset_indices=inset_idx,
        point_size=0.7, alpha=0.25, inset_zoom=0.40, fontscale=1.15,
    )

    # Panel labels
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.12, 1.10, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    # Shared legend
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=PC_COLORS[c],
               markeredgecolor="none", markersize=6,
               label=f"Class {c}")
        for c in range(pc_n_classes)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=pc_n_classes,
               fontsize=8, frameon=False, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(wspace=0.38)
    out = os.path.join(OUTPUT_DIR, "pc_scatter_4way_alexnet_fc1.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved standalone -> {out}")
    plt.close(fig)


def save_dense_pc_scatter(pretrained_pcs, pretrained_var, trained_pcs,
                          trained_var, pca_labels, pc_n_classes, img_paths,
                          n_images=1000, thumb_size=72, zoom=0.264,
                          figsize=(32, 15), dpi=300):
    """Save a high-density version where images replace scatter dots.

    Shows n_images total (uniformly sampled per class). Thumbnail size, zoom,
    figure size, and DPI are configurable for different density levels.
    """
    _thumb_cache.clear()
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Select n_images uniformly across classes
    rng = np.random.RandomState(42)
    per_class = n_images // pc_n_classes
    dense_idx = []
    for c in range(pc_n_classes):
        class_mask = np.where(pca_labels == c)[0]
        chosen = rng.choice(class_mask, size=min(per_class, len(class_mask)),
                            replace=False)
        dense_idx.extend(chosen.tolist())
    rng.shuffle(dense_idx)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, pcs, var_exp, title in [
        (axes[0], pretrained_pcs, pretrained_var, "Fine-grained (1000-way)"),
        (axes[1], trained_pcs, trained_var, f"Coarsened ({pc_n_classes}-way)"),
    ]:
        # Draw faint scatter for ALL points as background
        order = rng.permutation(len(pca_labels))
        for c in range(pc_n_classes):
            mask = pca_labels[order] == c
            ax.scatter(pcs[order[mask], 0], pcs[order[mask], 1],
                       c=PC_COLORS[c], s=0.3, alpha=0.15,
                       edgecolors="none", rasterized=True, zorder=1)

        # Overlay image thumbnails
        for ii in dense_idx:
            thumb = _get_thumbnail(img_paths[ii], size=thumb_size)
            if thumb is None:
                continue
            im_box = OffsetImage(thumb, zoom=zoom)
            im_box.image.axes = ax
            c = PC_COLORS[int(pca_labels[ii])]
            ab = AnnotationBbox(
                im_box, (pcs[ii, 0], pcs[ii, 1]),
                frameon=True, pad=0.03,
                bboxprops=dict(edgecolor=c, linewidth=1.0,
                               facecolor="none"),
                zorder=3,
            )
            ax.add_artist(ab)

        ax.set_xlabel("PC 1", fontsize=16, labelpad=5)
        ax.set_ylabel("PC 2", fontsize=16, labelpad=5)
        ax.set_title(title, fontsize=20, fontweight="bold", pad=12,
                     color="#1a1a1a")
        ax.tick_params(axis="both", labelsize=12, length=5, width=0.7)

        for idx in [0, 1]:
            lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
            margin = (hi - lo) * 0.05
            (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

        sns.despine(ax=ax, offset=4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    # Panel labels
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left")

    # Legend
    handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=PC_COLORS[c],
               markeredgecolor="none", markersize=12,
               label=f"Class {c}")
        for c in range(pc_n_classes)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=pc_n_classes,
               fontsize=14, frameon=False, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.subplots_adjust(wspace=0.25)
    ext = "jpg" if n_images >= 5000 else "png"
    out = os.path.join(OUTPUT_DIR, f"pc_scatter_dense_{n_images}.{ext}")
    save_kw = dict(dpi=dpi, bbox_inches="tight", facecolor="white",
                   edgecolor="none")
    if ext == "jpg":
        save_kw["pil_kwargs"] = {"quality": 92, "optimize": True}
    fig.savefig(out, **save_kw)
    print(f"Saved dense ({n_images} images) -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
