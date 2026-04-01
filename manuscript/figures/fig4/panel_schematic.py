"""Panel A: THINGS schematic — triplet task + pipeline diagram.

Layout (top to bottom):
  Title: "THINGS behavioral dataset"
  Row 1: Three THINGS images (airplane, cat, dog) — odd-one-out triplet task
  Row 2: Hand cursor (left) + behaviour stats (right) — what participants did
  Row 3: THINGS database mosaic (left) → Similarity embedding matrix (right)

Usage:
    Called by figure4.py as plot_schematic(ax).
"""

import os
import random

import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THINGS_ROOT = "/data/shared/datasets/hebart2019.things/images/object_images"
HAND_CURSOR_PATH = os.path.join(SCRIPT_DIR, "hand_cursor.png")

# ── Triplet config (airplane first so hand sits below dog on the right) ────
TRIPLET = [
    ("airplane", "airplane_01b.jpg"),
    ("cat",      "cat_01b.jpg"),
    ("dog",      "dog_01b.jpg"),
]
ODD_ONE_OUT = "airplane"
ODD_BORDER_COLOR = "#c44e52"
NORMAL_BORDER_COLOR = "#444444"

# ── Color palette ──────────────────────────────────────────────────────────
ARROW_COLOR = "#888888"
TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#555555"
TEXT_TERTIARY = "#777777"

# ── Embedding heatmap (10x16) ─────────────────────────────────────────────
EMBED_GRID = [
    ["#5fafa0","#e2ede6","#b0d2c2","#d2e0d8","#0a2640","#2a7a6a","#a0c8ba","#72b2a0","#388878","#125040"],
    ["#4a9a88","#88c0b0","#388878","#125040","#b8d5c8","#d5e2d8","#2a7062","#68ad95","#5aa588","#0e3a2e"],
    ["#ffffff","#1a6252","#d8e2da","#7eb8a2","#388a72","#b2d0c0","#5aa588","#0e3a2e","#98c5b2","#6aae92"],
    ["#ccdad2","#e5ebe5","#85bca5","#489a80","#98c5b2","#6aae92","#0d3045","#2a7868","#b0d2c2","#4a9a88"],
    ["#061e25","#358570","#1a6050","#70b092","#5fafa0","#b0d2c2","#4a9a88","#388878","#d2e0d8","#68ad95"],
    ["#72b2a0","#d2e0d8","#68ad95","#b8d5c8","#7eb8a2","#0e3a2e","#6aae92","#2a7a6a","#1a6252","#85bca5"],
    ["#b8d5c8","#388a72","#ccdad2","#0d3045","#72b2a0","#5fafa0","#e2ede6","#489a80","#2a7062","#358570"],
    ["#0a2640","#a0c8ba","#125040","#88c0b0","#d5e2d8","#1a6050","#70b092","#b2d0c0","#7eb8a2","#e5ebe5"],
    ["#388a72","#6aae92","#b0d2c2","#5fafa0","#2a7a6a","#ccdad2","#e2ede6","#85bca5","#d2e0d8","#1a6252"],
    ["#489a80","#0e3a2e","#72b2a0","#358570","#a0c8ba","#4a9a88","#d5e2d8","#2a7062","#125040","#b2d0c0"],
    ["#68ad95","#b8d5c8","#1a6050","#98c5b2","#88c0b0","#061e25","#7eb8a2","#70b092","#388878","#0d3045"],
    ["#e5ebe5","#5aa588","#d8e2da","#2a7868","#0a2640","#b0d2c2","#ffffff","#4a9a88","#6aae92","#d2e0d8"],
    ["#2a7062","#d2e0d8","#0e3a2e","#5fafa0","#72b2a0","#388a72","#ccdad2","#125040","#85bca5","#a0c8ba"],
    ["#1a6050","#489a80","#b8d5c8","#2a7a6a","#e2ede6","#68ad95","#358570","#88c0b0","#0d3045","#5aa588"],
    ["#70b092","#0a2640","#98c5b2","#4a9a88","#d5e2d8","#1a6252","#b2d0c0","#061e25","#7eb8a2","#6aae92"],
    ["#388878","#b0d2c2","#d8e2da","#e5ebe5","#2a7868","#ffffff","#0e3a2e","#72b2a0","#5fafa0","#ccdad2"],
]

MOSAIC_SEED = 42


def _load_image(concept, filename, size=256):
    """Load a THINGS concept image as a square crop."""
    path = os.path.join(THINGS_ROOT, concept, filename)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def _build_mosaic(n_cols=7, n_rows=7, tile_size=64):
    """Build a photo mosaic from random THINGS concept images."""
    rng = random.Random(MOSAIC_SEED)
    concepts = sorted(os.listdir(THINGS_ROOT))
    concepts = [c for c in concepts if os.path.isdir(os.path.join(THINGS_ROOT, c))]
    selected = rng.sample(concepts, min(n_cols * n_rows, len(concepts)))

    mosaic = np.zeros((n_rows * tile_size, n_cols * tile_size, 3), dtype=np.uint8)
    for idx, concept in enumerate(selected):
        row, col = divmod(idx, n_cols)
        if row >= n_rows:
            break
        concept_dir = os.path.join(THINGS_ROOT, concept)
        imgs = [f for f in os.listdir(concept_dir) if f.endswith(".jpg")]
        if not imgs:
            continue
        img_file = rng.choice(imgs)
        try:
            img = Image.open(os.path.join(concept_dir, img_file)).convert("RGB")
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            mosaic[row * tile_size:(row + 1) * tile_size,
                   col * tile_size:(col + 1) * tile_size] = np.array(img)
        except Exception:
            continue
    return mosaic


def _hex_grid_to_array(grid):
    """Convert a list-of-lists of hex colors to an (H, W, 3) float array."""
    rows = len(grid)
    cols = len(grid[0])
    arr = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            arr[r, c] = mcolors.to_rgb(grid[r][c])
    return arr


def plot_schematic(ax):
    """Draw the full THINGS schematic."""
    x_range, y_range = 10, 13
    ax.set_xlim(0, x_range)
    ax.set_ylim(0, y_range)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    ar = (ax_w_in / x_range) / (ax_h_in / y_range)

    # ══════════════════════════════════════════════════════════════════════
    # TITLE — use set_title so it aligns with panels B, C, D
    # ══════════════════════════════════════════════════════════════════════
    ax.set_title("THINGS behavioral dataset",
                 fontsize=11, fontweight="semibold", pad=8)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1: TRIPLET IMAGES (no rounded corners, no red label)
    # ══════════════════════════════════════════════════════════════════════
    img_w = 2.5
    img_h = img_w * ar
    img_gap = 0.45
    total_w = 3 * img_w + 2 * img_gap
    img_x_start = (x_range - total_w) / 2
    img_top = y_range - 0.55
    img_bottom = img_top - img_h

    for i, (concept, filename) in enumerate(TRIPLET):
        pil_img = _load_image(concept, filename)
        x_left = img_x_start + i * (img_w + img_gap)
        x_right = x_left + img_w

        ax.imshow(np.array(pil_img),
                  extent=[x_left, x_right, img_bottom, img_top],
                  aspect="auto", interpolation="lanczos", zorder=5)

        # Plain rectangular border — red for odd one out, gray for others
        is_odd = concept == ODD_ONE_OUT
        border_color = ODD_BORDER_COLOR if is_odd else NORMAL_BORDER_COLOR
        border_width = 2.5 if is_odd else 0.8
        border = plt.Rectangle((x_left, img_bottom), img_w, img_h,
                                fill=False, edgecolor=border_color,
                                linewidth=border_width, zorder=6)
        ax.add_patch(border)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2: BEHAVIOUR — hand icon (left) + stats (right)
    # ══════════════════════════════════════════════════════════════════════
    beh_top = img_bottom - 0.4
    beh_bottom = beh_top - 1.8
    beh_mid_y = (beh_top + beh_bottom) / 2

    # Hand cursor icon (smaller — 80% of previous)
    if os.path.exists(HAND_CURSOR_PATH):
        hand_img = Image.open(HAND_CURSOR_PATH).convert("RGBA")
        hand_arr = np.array(hand_img)
        hand_im = OffsetImage(hand_arr, zoom=0.072)
        hand_ab = AnnotationBbox(hand_im, (1.8, beh_mid_y),
                                  frameon=False, zorder=4)
        ax.add_artist(hand_ab)

    # Stats text (right of hand)
    stats_x = 3.5
    ax.text(stats_x, beh_mid_y + 0.4,
            "Odd-one-out similarity task",
            fontsize=9.5, fontweight="semibold", color=TEXT_PRIMARY,
            va="center", zorder=3)
    ax.text(stats_x, beh_mid_y - 0.1,
            "12,340 participants",
            fontsize=8.5, color=TEXT_SECONDARY, va="center", zorder=3)
    ax.text(stats_x, beh_mid_y - 0.55,
            "4.7 million similarity judgements",
            fontsize=8.5, color=TEXT_SECONDARY, va="center", zorder=3)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3: THINGS DATABASE (left) → SIMILARITY EMBEDDING (right)
    # ══════════════════════════════════════════════════════════════════════
    row3_top = beh_bottom - 0.15
    row3_bottom = 0.15
    row3_h = row3_top - row3_bottom
    row3_mid_y = (row3_top + row3_bottom) / 2

    # ── THINGS database mosaic (left) ──
    n_cols_m, n_rows_m = 7, 7
    mosaic_arr = _build_mosaic(n_cols=n_cols_m, n_rows=n_rows_m, tile_size=64)
    pixel_ar = mosaic_arr.shape[1] / mosaic_arr.shape[0]

    mosaic_w = 4.3
    mosaic_h = (mosaic_w / pixel_ar) * ar
    if mosaic_h > row3_h:
        mosaic_h = row3_h
        mosaic_w = mosaic_h * pixel_ar / ar
    mosaic_x = 0.4
    mosaic_y = row3_mid_y - mosaic_h / 2

    ax.imshow(mosaic_arr, extent=[mosaic_x, mosaic_x + mosaic_w,
                                   mosaic_y, mosaic_y + mosaic_h],
              aspect="auto", interpolation="lanczos", zorder=3)

    mosaic_frame = FancyBboxPatch((mosaic_x, mosaic_y), mosaic_w, mosaic_h,
                                   boxstyle="round,pad=0.0,rounding_size=0.12",
                                   fill=False, edgecolor="#888888",
                                   linewidth=0.8, zorder=5)
    ax.add_patch(mosaic_frame)

    # Dark overlay at bottom for label
    overlay_h = mosaic_h * 0.22
    overlay = plt.Rectangle((mosaic_x, mosaic_y), mosaic_w, overlay_h,
                              facecolor="black", alpha=0.55, zorder=4)
    ax.add_patch(overlay)
    ax.text(mosaic_x + mosaic_w / 2, mosaic_y + overlay_h / 2,
            "1,854 objects",
            fontsize=9, fontweight="bold", color="white",
            ha="center", va="center", zorder=6)

    # ── Arrow from mosaic to embedding ──
    arrow_x0 = mosaic_x + mosaic_w + 0.12
    arrow_x1 = arrow_x0 + 0.6
    ax.annotate("", xy=(arrow_x1, row3_mid_y + 0.3), xytext=(arrow_x0, row3_mid_y + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                lw=1.2, mutation_scale=12))

    # ── Similarity embedding (right) ──
    hm_w = 3.4
    hm_h = row3_h * 0.82
    hm_x = arrow_x1 + 0.35
    hm_y = row3_mid_y - hm_h / 2

    emb_arr = _hex_grid_to_array(EMBED_GRID)
    ax.imshow(emb_arr, extent=[hm_x, hm_x + hm_w, hm_y, hm_y + hm_h],
              aspect="auto", interpolation="nearest", zorder=3)
    hm_frame = plt.Rectangle((hm_x, hm_y), hm_w, hm_h,
                               fill=False, edgecolor="#aaaaaa",
                               linewidth=0.6, zorder=4)
    ax.add_patch(hm_frame)

    # Title above heatmap
    ax.text(hm_x + hm_w / 2, hm_y + hm_h + 0.15,
            "Similarity embedding",
            fontsize=9.5, fontweight="semibold", color=TEXT_PRIMARY,
            ha="center", va="bottom", zorder=3)

    # Axis labels
    ax.text(hm_x + hm_w / 2, hm_y - 0.1, "66 dimensions",
            fontsize=7, color=TEXT_TERTIARY, ha="center", va="top", zorder=3)
    ax.text(hm_x - 0.12, hm_y + hm_h / 2, "1,854 objects",
            fontsize=7, color=TEXT_TERTIARY, ha="right", va="center",
            rotation=90, zorder=3)
