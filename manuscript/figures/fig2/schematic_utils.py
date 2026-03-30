"""Schematic and brain-inset utilities for Figure 2.

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
ASSETS_DIR = Path("manuscript/figures/fig2/assets")
THINGS_IMG_DIR = Path("/data/shared/datasets/hebart2019.things/images/object_images")

# Stimuli selections (local PNGs in assets/)
TVSD_STIMULI = [
    ("guitar", "guitar_01b.jpg"),
    ("banana", "banana_01b.jpg"),
    ("butterfly", "butterfly_01b.jpg"),
    ("dog", "dog_01b.jpg"),
    ("hammer", "hammer_01b.jpg"),
]
NSD_STIMULI = [
    "nsd_beach_kite.png",
    "nsd_orange.png",
    "nsd_parrots.png",
    "nsd_horse_girl.png",
    "nsd_tennis.png",
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


def add_image_to_ax(ax, img, xy, zoom=0.12, border_color="#cccccc", border_width=0.8):
    """Place a PIL image on an axes as an AnnotationBbox with a border."""
    oi = OffsetImage(np.array(img), zoom=zoom)
    oi.image.axes = ax
    ab = AnnotationBbox(oi, xy, frameon=True, pad=0.15,
                        bboxprops=dict(edgecolor=border_color,
                                       linewidth=border_width,
                                       facecolor="white"))
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

def _place_svg_icon(ax, svg_path, xy, zoom, crop_box=None, flip_lr=False):
    """Load an SVG, optionally crop/flip, and place on axes."""
    img = load_svg_as_image(svg_path, width=400)
    if crop_box:
        w, h = img.size
        img = img.crop((int(w * crop_box[0]), int(h * crop_box[1]),
                         int(w * crop_box[2]), int(h * crop_box[3])))
    if flip_lr:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    oi = OffsetImage(np.array(img), zoom=zoom)
    ax.add_artist(AnnotationBbox(oi, xy, frameon=False))


def _draw_schematic_base(ax, stimuli_loader, stim_items,
                         species_svg, species_xy, species_zoom,
                         device_svg, device_xy, device_zoom, device_label,
                         species_crop=None, species_flip=False,
                         device_crop=None):
    """Shared layout: 5 images in horizontal strip → species icon + device."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 5 images in a horizontal strip
    n = len(stim_items)
    x_positions = np.linspace(0.04, 0.40, n)
    for x, item in zip(x_positions, stim_items):
        img = stimuli_loader(item, size=250)
        add_image_to_ax(ax, img, (x, 0.45), zoom=0.15,
                        border_color="#999999", border_width=1.2)

    # Arrow: images → species
    ax.annotate("", xy=(0.58, 0.45), xytext=(0.50, 0.45),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.18",
                                color="#666666", lw=2.5),
                zorder=5)

    # Species icon
    _place_svg_icon(ax, species_svg, species_xy, species_zoom,
                    crop_box=species_crop, flip_lr=species_flip)

    # Recording device icon
    _place_svg_icon(ax, device_svg, device_xy, device_zoom,
                    crop_box=device_crop)

    # Device label only
    ax.text(device_xy[0], 0.42, device_label, fontsize=6.5, ha="center",
            va="top", color="#777777", fontstyle="italic", zorder=10)


def draw_tvsd_schematic(ax):
    """TVSD schematic: object images → macaque brain + electrode array."""
    def _load(item, size):
        return load_things_image(item[0], item[1], size)

    _draw_schematic_base(
        ax,
        stimuli_loader=_load,
        stim_items=TVSD_STIMULI,
        species_svg=ASSETS_DIR / "macaque_brain.svg",
        species_xy=(0.74, 0.47),
        species_zoom=0.30,
        device_svg=ASSETS_DIR / "utah_array.svg",
        device_xy=(0.93, 0.22),
        device_zoom=0.20,
        device_label="Electrode\narray",
        device_crop=(0.05, 0.18, 0.75, 0.55),
    )


def draw_nsd_schematic(ax):
    """NSD schematic: natural scenes → human head + fMRI scanner."""
    def _load(item, size):
        return load_nsd_image(item, size)

    _draw_schematic_base(
        ax,
        stimuli_loader=_load,
        stim_items=NSD_STIMULI,
        species_svg=ASSETS_DIR / "human_head.svg",
        species_xy=(0.72, 0.50),
        species_zoom=0.32,
        device_svg=ASSETS_DIR / "mri_scanner.svg",
        device_xy=(0.92, 0.22),
        device_zoom=0.22,
        device_label="fMRI",
        species_crop=(0, 0, 1.0, 0.65),
        species_flip=True,
    )
