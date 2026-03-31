"""Figure 2b: Learned Representations — image mosaic at class centroids.

Two panels comparing 1000-way pretrained vs 4-way trained CNN representations,
each showing image thumbnails at per-class centroid positions in FC1 PCA space.

Usage (from project root):
    python manuscript/figures/fig2/plot_representations.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns

sys.path.insert(0, ".")

from manuscript.figures.fig2.utils import (
    SCRIPT_DIR, DATA_4WAY, INSET_LAYER, REPR_COLORS_4,
    setup_style, get_thumbnail, repel_positions,
    discrete_align_pcs, extract_class_indices,
    compute_centroids, closest_image_to_centroid,
)

N_SAMPLE = 1000  # number of individual images to show as thumbnails
THUMB_ZOOM = 0.154  # 10% larger than original 0.14

# Inset: square, top-left corner, axes-fraction coords [x0, y0, w, h]
INSET_RECT = [0.01, 0.72, 0.24, 0.24]


def sample_images(pcs, n_sample, seed=42):
    """Randomly sample n_sample image indices from the full dataset."""
    rng = np.random.RandomState(seed)
    return rng.choice(len(pcs), size=min(n_sample, len(pcs)), replace=False)


def _in_inset_region(x, y, xlim, ylim):
    """Check if a point (in data coords) falls inside the inset rectangle + margin."""
    x_frac = (x - xlim[0]) / (xlim[1] - xlim[0])
    y_frac = (y - ylim[0]) / (ylim[1] - ylim[0])
    x0, y0, w, h = INSET_RECT
    return x_frac < (x0 + w + 0.04) and y_frac > (y0 - 0.04)


def plot_mosaic_panel(ax, pcs, labels_4way, colors, title,
                      img_paths, sample_idx,
                      thumb_size=96, thumb_zoom=THUMB_ZOOM, show_ylabel=True):
    """Image mosaic of sampled images at actual PC coords + square dot inset."""
    # Axis limits from all points
    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.08
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Image thumbnails, skip those under the inset
    for ii in sample_idx:
        if _in_inset_region(pcs[ii, 0], pcs[ii, 1], xlim, ylim):
            continue
        thumb = get_thumbnail(str(img_paths[ii]), size=thumb_size)
        if thumb is None:
            continue
        im_box = OffsetImage(thumb, zoom=thumb_zoom)
        im_box.image.axes = ax
        ab = AnnotationBbox(im_box, (pcs[ii, 0], pcs[ii, 1]),
                            frameon=False, pad=0, zorder=6)
        ax.add_artist(ab)

    # Main axis formatting
    ax.set_xlabel("PC 1", fontsize=10, labelpad=5, style="italic")
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=10, labelpad=5, style="italic")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)
    sns.despine(ax=ax, offset=5, left=not show_ylabel)

    # Square dot-scatter inset (top-left, white background)
    inset = ax.inset_axes(INSET_RECT)
    inset.set_facecolor("white")
    rng = np.random.RandomState(42)
    order = rng.permutation(len(pcs))
    point_colors = [colors[labels_4way[i] % len(colors)] for i in order]
    inset.scatter(pcs[order, 0], pcs[order, 1],
                  c=point_colors, s=0.4, alpha=0.30,
                  edgecolors="none", rasterized=True)
    # Match aspect ratio of data to keep it square
    inset.set_xlim(xlim)
    inset.set_ylim(ylim)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")


def plot_representations(save=True):
    """Generate Figure 2b: learned representation mosaic."""
    setup_style()

    if not os.path.exists(DATA_4WAY):
        print(f"ERROR: 4-way data not found at {DATA_4WAY}")
        print("Run: python experiments/representation_analysis/2pcs_compare/run_analysis.py")
        return None

    d4 = np.load(DATA_4WAY, allow_pickle=True)
    labels_4way = d4["pca_labels"]
    img_paths = d4["img_paths"]
    pcs_4way_trained = d4[f"{INSET_LAYER}_trained_pcs"].copy()
    pcs_pretrained = d4[f"{INSET_LAYER}_pretrained_pcs"].copy()

    pcs_pretrained_aligned = discrete_align_pcs(
        pcs_pretrained, pcs_4way_trained, labels_4way, 4)

    # Sample the same 1,000 images for both panels
    sample_idx = sample_images(pcs_4way_trained, N_SAMPLE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_mosaic_panel(axes[0], pcs_pretrained_aligned, labels_4way,
                      REPR_COLORS_4, "",
                      img_paths, sample_idx,
                      show_ylabel=True)

    plot_mosaic_panel(axes[1], pcs_4way_trained, labels_4way,
                      REPR_COLORS_4, "",
                      img_paths, sample_idx,
                      show_ylabel=False)

    plt.tight_layout(w_pad=2.0)

    # Panel labels + titles
    for ax, label, title in [
        (axes[0], "a", "CNN trained on 1,000 classes"),
        (axes[1], "b", "CNN trained on 4 coarse classes"),
    ]:
        ax.text(-0.02, 1.08, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")
        ax.set_title(title, fontsize=12, fontweight="normal", pad=10,
                     color="#1a1a1a")

    if save:
        out = os.path.join(SCRIPT_DIR, "figure2.png")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                    edgecolor="none")
        print(f"Saved -> {out}")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    plot_representations()
