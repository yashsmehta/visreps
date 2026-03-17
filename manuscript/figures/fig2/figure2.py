"""Figure 2: Coarse Representations Are Fundamentally Different.

Composite figure:
  A — 2×3 grid of class-level RDMs (4,8,16,32,64,1000-way CLIP-PCA)
  B — PC1/PC2 scatter: fine-grained (top) vs coarse (bottom) with image insets

Usage:
    python manuscript/figures/fig2/figure2.py
    python manuscript/figures/fig2/figure2.py --recompute-rdms
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
import seaborn as sns
from scipy.stats import rankdata

import torch

sys.path.insert(0, ".")
from visreps.analysis.rsa import compute_rdm

# Import shared RDM helpers from plot_class_rdms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_class_rdms import (
    COARSE_CFG_IDS, rank_transform,
    build_sort_order, draw_sidebar, draw_boundaries,
    CATEGORY_NAMES, CATEGORY_COLORS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
RDM_CACHE = os.path.join(SCRIPT_DIR, "class_rdm_data.npz")
PC_DATA = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                       "2pcs_compare", "data_4way_alexnet.npz")
OUTPUT_DIR = SCRIPT_DIR


# ── RDM panel (rank-transformed for composite figure) ───────────────────

def plot_rdm_panel(ax, rdm, title, block_boundaries=None):
    """Draw a single RDM panel with rank-transformed dissimilarity."""
    rdm_ranked = rank_transform(rdm)
    n = rdm.shape[0]
    im = ax.imshow(rdm_ranked, cmap="magma", interpolation="nearest",
                   aspect="equal", rasterized=True, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7, color="#1a1a1a")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if block_boundaries is not None:
        draw_boundaries(ax, block_boundaries, n)
        draw_sidebar(ax, block_boundaries, n, side="left")
        draw_sidebar(ax, block_boundaries, n, side="bottom")

    return im


# ── Panel B — PC scatter with image insets ────────────────────────────────

PC_COLORS = ["#00A896", "#7B68EE", "#E8963E", "#D64045"]
N_INSETS = 3      # images per class
INSET_LAYER = "fc1"


def align_pc_projections(pretrained_pcs, trained_pcs, labels, n_classes):
    """Align the pretrained PCA projection to best match the trained one
    using Procrustes rotation on class centroids.

    This finds the optimal 2D rotation matrix R that minimizes the distance
    between class centroids: ||pre_centroids @ R - tra_centroids||.
    Unlike axis flips or 90° swaps, Procrustes preserves the PC axis identity
    (PC1 stays PC1) while rotating the point cloud for visual alignment.

    Returns (aligned_pretrained_pcs, trained_pcs).
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

    return aligned_pcs, trained_pcs


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


def plot_pc_scatter_panel(ax, pcs, labels, n_classes, colors, title,
                          img_paths=None, inset_indices=None,
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
    ax.set_xlabel("PC 1", fontsize=8.5 * fs, labelpad=3)
    ax.set_ylabel("PC 2", fontsize=8.5 * fs, labelpad=3)
    ax.set_title(title, fontsize=10 * fs, fontweight="bold", pad=6,
                 color="#1a1a1a")
    ax.tick_params(axis="both", labelsize=7 * fs, length=2.5, width=0.5, pad=2)

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.12
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=4)
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
                bboxprops=dict(edgecolor=c, linewidth=2.0,
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

    sns.set_theme(style="ticks", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ── Load RDM data ──
    data = np.load(RDM_CACHE)
    centroids_1k = data["centroids_1k"]
    centroids_pre = data["centroids_pretrained"]
    n_classes = centroids_1k.shape[0]

    # Sort order: 11-category WordNet scheme, within-category clustering
    # from seed A (seed=1) 1000-way model (independent from displayed seed C)
    categories = data["categories"]
    seed_a_cache = os.path.join(SCRIPT_DIR, "centroids_1k_seedA.npz")
    if os.path.exists(seed_a_cache):
        centroids_sort = np.load(seed_a_cache)["centroids"]
    else:
        print("Warning: seed A centroids not found, falling back to pretrained.")
        centroids_sort = centroids_pre
    rdm_sort = compute_rdm(torch.tensor(centroids_sort, dtype=torch.float32)).numpy()
    sort_idx, block_boundaries = build_sort_order(categories, rdm_sort)
    rdm_1k = compute_rdm(torch.tensor(centroids_1k, dtype=torch.float32)).numpy()
    rdm_1k_sorted = rdm_1k[np.ix_(sort_idx, sort_idx)]

    coarse_rdms = {}
    for cfg_id in COARSE_CFG_IDS:
        key = f"centroids_{cfg_id}"
        if key in data:
            cent = data[key]
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
        trained_pcs = pc_data[f"{INSET_LAYER}_trained_pcs"].copy()
        # Align projections: rotate pretrained to match trained via Procrustes
        pretrained_pcs, trained_pcs = align_pc_projections(
            pretrained_pcs, trained_pcs, pca_labels, pc_n_classes)
        # Select inset images from coarse panel (clear clusters), show in both
        inset_idx = _select_inset_indices(trained_pcs, pca_labels, pc_n_classes)
    else:
        print("Warning: PC scatter data not found, Panel B will be skipped.")

    # ── Figure layout ──
    fig = plt.figure(figsize=(15, 8.5))

    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.15, 0.55],
        wspace=0.18,
        left=0.02, right=0.97, top=0.93, bottom=0.04,
    )

    # ── Panel A: 2×2 RDM grid (equal-sized subplots) ──
    gs_rdms = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0, 0],
        wspace=0.12, hspace=0.18,
    )

    rdm_order = [4, 16, 64, 1000]
    rdm_titles = ["4-way", "16-way", "64-way", "1000-way"]
    rdm_axes = []
    im = None

    for i, (cfg_id, title) in enumerate(zip(rdm_order, rdm_titles)):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs_rdms[row, col])
        rdm_axes.append(ax)
        rdm = rdm_1k_sorted if cfg_id == 1000 else coarse_rdms.get(cfg_id)
        if rdm is not None:
            im = plot_rdm_panel(ax, rdm, title, block_boundaries)
        else:
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6,
                         color="#1a1a1a")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999")
            ax.set_xticks([]); ax.set_yticks([])
        # Ensure square aspect ratio
        ax.set_aspect("equal")

    # Colorbar — inset next to last RDM in top row
    ax_last_top = rdm_axes[1]  # top-right RDM
    ax_cb = inset_axes(ax_last_top, width="3.5%", height="100%",
                       loc="center right",
                       bbox_to_anchor=(0.09, 0, 1, 1),
                       bbox_transform=ax_last_top.transAxes, borderpad=0)
    cb = plt.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=7, length=2.5, width=0.5, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
    cb.ax.set_ylabel("Dissimilarity (rank)", fontsize=7.5, labelpad=7,
                      rotation=270, va="bottom")

    # ── Category legend for RDM sidebars ──
    from matplotlib.patches import Patch
    cat_handles = [
        Patch(facecolor=CATEGORY_COLORS[i], edgecolor="none",
              label=CATEGORY_NAMES[i])
        for i in range(len(CATEGORY_NAMES))
    ]
    # Place below the bottom-left RDM
    rdm_axes[2].legend(
        handles=cat_handles, loc="upper center",
        bbox_to_anchor=(1.55, -0.08), ncol=6,
        fontsize=6.5, frameon=False, handletextpad=0.3,
        columnspacing=0.8, handlelength=1.2,
    )

    # ── Panel B: PC scatter with image insets ──
    if has_pc_data:
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_outer[0, 1],
            height_ratios=[1.0, 1.0],
            hspace=0.42,
        )
        ax_pc_pre = fig.add_subplot(gs_right[0, 0])
        ax_pc_coarse = fig.add_subplot(gs_right[1, 0])

        # Same images shown in both panels so viewer sees reorganization
        plot_pc_scatter_panel(
            ax_pc_pre, pretrained_pcs,
            pca_labels, pc_n_classes, PC_COLORS,
            title="Fine-grained (1000-way)",
            img_paths=img_paths, inset_indices=inset_idx,
            point_size=0.8, alpha=0.30, inset_zoom=0.42,
        )
        plot_pc_scatter_panel(
            ax_pc_coarse, trained_pcs,
            pca_labels, pc_n_classes, PC_COLORS,
            title=f"Coarsened ({pc_n_classes}-way)",
            img_paths=img_paths, inset_indices=inset_idx,
            point_size=0.8, alpha=0.30, inset_zoom=0.42,
        )

        # Shared legend for scatter panels
        scatter_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PC_COLORS[c],
                   markeredgecolor="none", markersize=6,
                   label=f"Class {c}")
            for c in range(pc_n_classes)
        ]
        ax_pc_coarse.legend(
            handles=scatter_handles, loc="lower center",
            bbox_to_anchor=(0.5, -0.22), ncol=pc_n_classes,
            fontsize=8, frameon=False, handletextpad=0.2,
            columnspacing=0.8,
        )

    # ── Panel labels ──
    rdm_axes[0].text(-0.08, 1.10, "A", transform=rdm_axes[0].transAxes,
                     fontsize=16, fontweight="bold", va="top", ha="left",
                     fontfamily="sans-serif")
    if has_pc_data:
        ax_pc_pre.text(-0.14, 1.12, "B", transform=ax_pc_pre.transAxes,
                       fontsize=16, fontweight="bold", va="top", ha="left",
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
            pretrained_pcs, trained_pcs,
            pca_labels, pc_n_classes, img_paths, inset_idx,
        )
        save_dense_pc_scatter(
            pretrained_pcs, trained_pcs,
            pca_labels, pc_n_classes, img_paths, n_images=1000,
        )


def save_standalone_pc_scatter(pretrained_pcs, trained_pcs, pca_labels,
                               pc_n_classes, img_paths, inset_idx):
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
        axes[0], pretrained_pcs,
        pca_labels, pc_n_classes, PC_COLORS,
        title="Fine-grained (1000-way)",
        img_paths=img_paths, inset_indices=inset_idx,
        point_size=0.7, alpha=0.25, inset_zoom=0.40, fontscale=1.15,
    )
    plot_pc_scatter_panel(
        axes[1], trained_pcs,
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


def save_dense_pc_scatter(pretrained_pcs, trained_pcs, pca_labels,
                          pc_n_classes, img_paths,
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

    for ax, pcs, title in [
        (axes[0], pretrained_pcs, "Fine-grained (1000-way)"),
        (axes[1], trained_pcs, f"Coarsened ({pc_n_classes}-way)"),
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
