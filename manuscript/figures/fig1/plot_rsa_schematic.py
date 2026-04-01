"""RSA schematic — two 6×6 similarity matrices for Figure 1.

Illustrative (fake but plausible values) model and neural RDMs using the
same 6 ImageNet classes shown as insets in Figure 1a.

Usage (from project root):
    python manuscript/figures/fig1/plot_rsa_schematic.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colorbar import ColorbarBase

sys.path.insert(0, ".")

from manuscript.figures.fig1.utils import (
    INSET_CLASSES, setup_style, _get_inset_image,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Concepts (same 6 as fig1a insets) ─────────────────────────────────
# Order: armchair, barber chair, school bus, African violet, wood frog, bullfrog
CONCEPT_NAMES = ["armchair", "barber chair", "school bus",
                 "African violet", "wood frog", "bullfrog"]

# ── Fake similarity matrices ──────────────────────────────────────────
# Uneven, visually varied values — purely schematic, no semantic logic.

MODEL_SIM = np.array([
    [1.00, 0.58, 0.74, 0.21, 0.47, 0.33],
    [0.58, 1.00, 0.39, 0.65, 0.14, 0.82],
    [0.74, 0.39, 1.00, 0.50, 0.28, 0.61],
    [0.21, 0.65, 0.50, 1.00, 0.73, 0.16],
    [0.47, 0.14, 0.28, 0.73, 1.00, 0.44],
    [0.33, 0.82, 0.61, 0.16, 0.44, 1.00],
])

NEURAL_SIM = np.array([
    [1.00, 0.45, 0.68, 0.30, 0.55, 0.19],
    [0.45, 1.00, 0.22, 0.71, 0.38, 0.63],
    [0.68, 0.22, 1.00, 0.41, 0.76, 0.34],
    [0.30, 0.71, 0.41, 1.00, 0.52, 0.80],
    [0.55, 0.38, 0.76, 0.52, 1.00, 0.26],
    [0.19, 0.63, 0.34, 0.80, 0.26, 1.00],
])

THUMB_PX = 96


def load_inset_images(size=THUMB_PX):
    """Load the 6 ImageNet inset images used in fig1a."""
    from dotenv import load_dotenv
    load_dotenv()
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR", "")

    images = {}
    for i, (cls_idx, synset_id, img_idx) in enumerate(INSET_CLASSES):
        thumb = _get_inset_image(synset_id, imagenet_dir, size=size,
                                 image_index=img_idx)
        if thumb is not None:
            images[i] = thumb
        else:
            print(f"WARNING: Could not load image for {CONCEPT_NAMES[i]}"
                  f" ({synset_id})")
    return images


def _blue_cmap():
    """Blue sequential colormap for model RDM."""
    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    return mcolors.LinearSegmentedColormap.from_list("rdm_blue", colors, N=256)


def _purple_cmap():
    """Purple sequential colormap for neural RDM."""
    colors = ["#fcfbfd", "#d2d0e7", "#9e9ac8", "#6a51a3", "#3f007d"]
    return mcolors.LinearSegmentedColormap.from_list("rdm_purple", colors, N=256)


def draw_rdm(sim_matrix, cmap, cbar_label, filename, images):
    """Draw a single lower-triangular RDM with concept images on both axes."""
    n = len(CONCEPT_NAMES)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Mask upper triangle (keep diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    masked = np.ma.array(sim_matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    fig.patch.set_facecolor("white")

    # Draw matrix
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal",
              interpolation="nearest", zorder=1)

    # White grid lines for cell boundaries
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2, zorder=2)
        ax.axvline(i - 0.5, color="white", linewidth=2, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Concept thumbnails ──
    img_zoom = 0.38
    offset = 1.05

    bbox_props = dict(
        edgecolor="#dddddd", linewidth=0.4,
        facecolor="white", boxstyle="square,pad=0.04",
    )

    for i in range(n):
        if i not in images:
            continue
        thumb = images[i]

        # Top axis
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=img_zoom),
            (i, -offset), frameon=True, pad=0.08,
            bboxprops=bbox_props, box_alignment=(0.5, 0.5),
            zorder=10, xycoords="data", clip_on=False,
        ))

        # Left axis
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=img_zoom),
            (-offset, i), frameon=True, pad=0.08,
            bboxprops=bbox_props, box_alignment=(0.5, 0.5),
            zorder=10, xycoords="data", clip_on=False,
        ))

    # Axis limits with room for thumbnails
    ax.set_xlim(-1.8, n - 0.5 + 0.15)
    ax.set_ylim(n - 0.5 + 0.15, -1.8)

    # ── Colorbar ──
    cax = fig.add_axes([0.82, 0.28, 0.025, 0.40])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(cbar_label, fontsize=12, labelpad=10)
    cb.ax.tick_params(labelsize=10, length=3, width=0.4)
    cb.outline.set_linewidth(0.3)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    plt.subplots_adjust(left=0.12, right=0.78, top=0.94, bottom=0.08)

    out = os.path.join(SCRIPT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close(fig)


def main():
    setup_style()
    images = load_inset_images()

    draw_rdm(MODEL_SIM, _blue_cmap(), "Similarity",
             "rsa_schematic_model.png", images)
    draw_rdm(NEURAL_SIM, _purple_cmap(), "Similarity",
             "rsa_schematic_neural.png", images)


if __name__ == "__main__":
    main()
