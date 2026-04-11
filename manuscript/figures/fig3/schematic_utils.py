"""Schematic and brain-inset utilities for Figure 3.

Provides:
- SVG/image loading helpers (load_svg_as_image, load_things_image, load_nsd_image)
- Brain inset rendering (add_brain_inset) using nilearn PNGs for human, SVG+ellipse for macaque
- Dataset schematic drawing (draw_tvsd_schematic, draw_nsd_schematic)

Brain region PNGs are pre-rendered by generate_brain_renders.py and stored in assets/.
"""

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from PIL import Image

# ── Asset paths ─────────────────────────────────────────────────────────
ASSETS_DIR = Path("manuscript/figures/fig3/assets")
THINGS_IMG_DIR = Path("/data/shared/datasets/hebart2019.things/images/object_images")

# Stimuli selections (local PNGs in assets/)
TVSD_STIMULI = [
    ("guitar", "guitar_01b.jpg"),
    ("banana", "banana_01b.jpg"),
    ("butterfly", "butterfly_01b.jpg"),
    ("dog", "dog_01b.jpg"),
    ("hammer", "hammer_01b.jpg"),
    ("flower", "flower_01b.jpg"),
]
NSD_STIMULI = [
    "nsd_beach_kite.png",
    "nsd_orange.png",
    "nsd_parrots.png",
    "nsd_horse_girl.png",
    "nsd_tennis.png",
    "nsd_cat_bucket.png",
]

# Brain region highlight style
REGION_HIGHLIGHT = "#e74c3c"
REGION_HIGHLIGHT_ALPHA = 0.55

# Pre-rendered nilearn brain PNGs (human)
_NILEARN_RENDERS = {
    "early":   ASSETS_DIR / "brain_early_visual.png",
    "ventral": ASSETS_DIR / "brain_ventral_visual.png",
}

# Macaque ROI ellipse positions (tuned to Scidraw lateral-view SVG)
_MACAQUE_ROI_POS = {
    "V1": {"xy": (0.18, 0.48), "width": 0.15, "height": 0.18},
    "IT": {"xy": (0.72, 0.30), "width": 0.16, "height": 0.15},
}


# ── Image loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def load_svg_as_image(svg_path, width=300):
    """Render an SVG file to a PIL RGBA Image (cached)."""
    png_data = cairosvg.svg2png(url=str(svg_path), output_width=width, unsafe=True)
    return Image.open(io.BytesIO(png_data)).convert("RGBA")


def load_things_image(concept, filename, size=120):
    """Load a THINGS object image, center-crop to square, resize."""
    img = Image.open(THINGS_IMG_DIR / concept / filename).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def load_nsd_image(filename, size=120):
    """Load a pre-saved NSD stimulus PNG from assets, resize to square."""
    return Image.open(ASSETS_DIR / filename).convert("RGB").resize(
        (size, size), Image.LANCZOS)


def add_image_to_ax(ax, img, xy, zoom=0.12, border_color="#cccccc",
                    border_width=0.8, zorder=3):
    """Place a PIL image on an axes as an AnnotationBbox with an optional border."""
    oi = OffsetImage(np.array(img), zoom=zoom)
    oi.image.axes = ax
    frameon = border_width > 0
    ab = AnnotationBbox(oi, xy, frameon=frameon, pad=0.15 if frameon else 0,
                        bboxprops=dict(edgecolor=border_color,
                                       linewidth=border_width,
                                       facecolor="white") if frameon else None)
    ab.set_zorder(zorder)
    ax.add_artist(ab)
    return ab


# ── Brain insets ────────────────────────────────────────────────────────

def add_brain_inset(ax, brain_type, region, inset_bounds=(0.65, 0.58, 0.32, 0.38)):
    """Add a small brain inset with highlighted ROI to a data panel.

    brain_type: 'macaque' or 'human'
    region: 'V1', 'IT' (macaque) or 'early', 'ventral' (human)
    """
    inset = ax.inset_axes(inset_bounds)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.axis("off")

    bg = mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.03",
        facecolor="white", edgecolor="#dddddd",
        linewidth=0.4, alpha=0.88, zorder=0)
    inset.add_patch(bg)

    if brain_type == "human" and region in _NILEARN_RENDERS:
        brain_arr = np.array(Image.open(_NILEARN_RENDERS[region]).convert("RGBA"))
        inset.imshow(brain_arr, extent=[0.02, 0.98, 0.02, 0.98],
                     aspect="auto", zorder=1)
    else:
        brain_arr = np.array(
            load_svg_as_image(ASSETS_DIR / "macaque_brain.svg", width=200))
        inset.imshow(brain_arr, extent=[0.02, 0.98, 0.08, 0.92],
                     aspect="auto", zorder=1)
        if region in _MACAQUE_ROI_POS:
            pos = _MACAQUE_ROI_POS[region]
            inset.add_patch(mpatches.Ellipse(
                pos["xy"], pos["width"], pos["height"],
                facecolor=REGION_HIGHLIGHT, alpha=REGION_HIGHLIGHT_ALPHA,
                edgecolor=REGION_HIGHLIGHT, linewidth=0.8, zorder=2))

    return inset


# ── Schematic panels ────────────────────────────────────────────────────

def _place_png_icon(ax, png_path, xy, zoom, flip_lr=False, zorder=6):
    """Load a PNG and place on axes without a border."""
    img = Image.open(png_path).convert("RGBA")
    if flip_lr:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    add_image_to_ax(ax, img, xy, zoom=zoom, border_color="none",
                    border_width=0, zorder=zorder)


def _draw_image_grid(ax, stimuli_loader, stim_items, grid_center,
                     grid_width=0.30, grid_height=0.48, img_zoom=0.11,
                     rows=2, cols=3):
    """Draw a grid of stimulus images centered at grid_center.

    rows=1 or cols=1 degenerate correctly (linspace returns midpoint).
    """
    cx, cy = grid_center
    xs = [cx] if cols == 1 else np.linspace(cx - grid_width / 2, cx + grid_width / 2, cols)
    ys = [cy] if rows == 1 else np.linspace(cy + grid_height / 2, cy - grid_height / 2, rows)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(stim_items):
                img = stimuli_loader(stim_items[idx], size=250)
                add_image_to_ax(ax, img, (xs[c], ys[r]), zoom=img_zoom,
                                border_color="#cccccc", border_width=0.5,
                                zorder=4)
                idx += 1


def _draw_schematic_base(ax, stimuli_loader, stim_items, icons,
                         arrow_xytext=None, arrow_xy=None,
                         stats_lines=None, caption=None, caption_xy=None,
                         headline=None, headline_xy=None,
                         grid_center=(0.18, 0.50),
                         grid_width=0.28, grid_height=0.46,
                         img_zoom=0.12, rows=3, cols=2):
    """Shared layout: tight image grid → right-pointing arrow → icon(s).

    The arrow terminates *before* the first icon so its head is fully
    visible (no zorder games). Stats are drawn in the top-right of the axes.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_image_grid(ax, stimuli_loader, stim_items,
                     grid_center=grid_center, grid_width=grid_width,
                     grid_height=grid_height, img_zoom=img_zoom,
                     rows=rows, cols=cols)

    if arrow_xytext is not None and arrow_xy is not None:
        ax.annotate("", xy=arrow_xy, xytext=arrow_xytext,
                    arrowprops=dict(arrowstyle="->", color="#a8a8a8",
                                    lw=0.95, shrinkA=0, shrinkB=0,
                                    mutation_scale=11),
                    zorder=10)

    for png_path, xy, zoom, flip_lr in icons:
        _place_png_icon(ax, png_path, xy, zoom, flip_lr=flip_lr, zorder=6)

    if headline and headline_xy is not None:
        ax.text(headline_xy[0], headline_xy[1], headline,
                fontsize=12.5, color="#888888", style="italic",
                ha="center", va="top", transform=ax.transAxes)

    if caption and caption_xy is not None:
        ax.text(caption_xy[0], caption_xy[1], caption,
                fontsize=11.25, color="#555555", style="italic",
                ha="center", va="top", transform=ax.transAxes)

    if stats_lines:
        # Align stats block top with headline baseline so the left "image count"
        # line and the right "subjects/regions" first line sit in the same row.
        stats_y = headline_xy[1] if headline_xy is not None else 0.97
        ax.text(0.99, stats_y, "\n".join(stats_lines),
                fontsize=12.5, color="#888888", style="italic",
                ha="right", va="top", linespacing=1.35,
                transform=ax.transAxes)


def draw_tvsd_schematic(ax):
    """TVSD schematic: object images → macaque icon."""
    def _load(item, size):
        return load_things_image(item[0], item[1], size)

    _draw_schematic_base(
        ax, _load, TVSD_STIMULI,
        icons=[(ASSETS_DIR / "monkey.png", (0.76, 0.48), 0.083, False)],
        arrow_xytext=(0.45, 0.48), arrow_xy=(0.60, 0.48),
        stats_lines=[
            "2 macaques",
            "V1, V4, IT recordings",
        ],
        headline="22,248\nobject images",
        headline_xy=(0.16, 0.99),
        caption="Macaque\nelectrophysiology",
        caption_xy=(0.76, 0.26),
        grid_center=(0.16, 0.48), grid_width=0.15, grid_height=0.40,
        img_zoom=0.2112, rows=3, cols=2,
    )


def draw_nsd_schematic(ax):
    """NSD schematic: natural scenes → human head + fMRI scanner."""
    def _load(item, size):
        return load_nsd_image(item, size)

    _draw_schematic_base(
        ax, _load, NSD_STIMULI,
        icons=[
            (ASSETS_DIR / "human.png", (0.64, 0.42), 0.077, False),
            (ASSETS_DIR / "fmri.png", (0.85, 0.42), 0.083, False),
        ],
        arrow_xytext=(0.40, 0.42), arrow_xy=(0.52, 0.42),
        stats_lines=[
            "8 human subjects",
            "ventral cortex",
        ],
        headline="73,000\nnatural scenes",
        headline_xy=(0.16, 0.97),
        caption="Human\n7T fMRI",
        caption_xy=(0.745, 0.20),
        grid_center=(0.16, 0.44), grid_width=0.15, grid_height=0.40,
        img_zoom=0.2112, rows=3, cols=2,
    )


