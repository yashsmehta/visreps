"""Shared utilities for THINGS visualizations."""

import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import umap

# ── Paths & sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "things_visualizations", "data", "things_viz_data.npz")
FIG_DIR = os.path.join(PROJECT_ROOT, "experiments", "things_visualizations", "figures")

# Ensure project root is on sys.path so scripts can import visreps
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_data():
    """Load the pre-extracted THINGS visualization data."""
    return np.load(DATA_PATH, allow_pickle=True)


# ── UMAP ──────────────────────────────────────────────────────────────

def run_umap(features, n_neighbors=50, min_dist=0.2, pca_dim=50, seed=42):
    """Z-score, PCA pre-reduction, then UMAP to 2D."""
    mu = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    features = (features - mu) / std

    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric="euclidean", random_state=seed, verbose=False,
    )
    return reducer.fit_transform(features.astype(np.float32))


def grid_subsample(coords, grid_size):
    """Select concept indices for even spatial coverage via grid-based sampling.

    Divides the 2D space into grid_size x grid_size cells and picks
    the nearest point to each cell center. Returns unique sorted indices.
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.08
    pad_y = (y_max - y_min) * 0.08

    x_centers = np.linspace(x_min + pad_x, x_max - pad_x, grid_size)
    y_centers = np.linspace(y_min + pad_y, y_max - pad_y, grid_size)

    selected = set()
    for xc in x_centers:
        for yc in y_centers:
            dists = (coords[:, 0] - xc) ** 2 + (coords[:, 1] - yc) ** 2
            selected.add(int(np.argmin(dists)))
    return np.array(sorted(selected))


def setup_umap_axes(ax, coords, title, bg="#fafafa", scatter=True, pad_frac=0.12):
    """Configure a matplotlib axes for UMAP scatter display."""
    ax.set_facecolor(bg)
    if scatter:
        ax.scatter(coords[:, 0], coords[:, 1], c="#d0d0d0", s=4, alpha=0.5,
                   edgecolors="none", rasterized=True, zorder=1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    x_pad = (coords[:, 0].max() - coords[:, 0].min()) * pad_frac
    y_pad = (coords[:, 1].max() - coords[:, 1].min()) * pad_frac
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Thumbnails ────────────────────────────────────────────────────────

def load_thumbnail(path, size=64, border_width=2, border_color=(60, 60, 60)):
    """Load an image, resize it, and add a border."""
    try:
        img = Image.open(path).convert("RGB")
        inner = size - 2 * border_width
        img = img.resize((inner, inner), Image.LANCZOS)
        bordered = Image.new("RGB", (size, size), border_color)
        bordered.paste(img, (border_width, border_width))
        return bordered
    except Exception:
        return Image.new("RGB", (size, size), (128, 128, 128))


def coords_to_pixels(coords, canvas_size, thumb_size, padding_frac=0.04):
    """Map UMAP coordinates to pixel positions on a canvas."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range, y_range = x_max - x_min, y_max - y_min
    pad_x, pad_y = x_range * padding_frac, y_range * padding_frac

    half = thumb_size // 2
    usable = canvas_size - thumb_size

    px = ((coords[:, 0] - x_min + pad_x) / (x_range + 2 * pad_x) * usable + half).astype(int)
    py = ((coords[:, 1] - y_min + pad_y) / (y_range + 2 * pad_y) * usable + half).astype(int)
    py = canvas_size - py  # flip y
    return px, py


# ── PIL panel helpers ─────────────────────────────────────────────────

def get_font(size):
    """Load a bold font at the given size, with fallbacks."""
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_frame(panel_img, width=2, color=(160, 160, 160)):
    """Add a thin frame border around a PIL image."""
    w, h = panel_img.size
    framed = Image.new("RGB", (w + 2 * width, h + 2 * width), color)
    framed.paste(panel_img, (width, width))
    return framed


def add_title(panel_img, title, height=110, font_size=56, bg=(255, 255, 255)):
    """Add a centered title above a PIL panel image."""
    w, h = panel_img.size
    full = Image.new("RGB", (w, h + height), bg)
    full.paste(panel_img, (0, height))

    draw = ImageDraw.Draw(full)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (height - th) // 2 + 10), title, fill=(20, 20, 20), font=font)
    return full


def stitch_panels(panels, gap=100, margin=50, bg=(255, 255, 255)):
    """Stitch PIL panels side by side with gap and margin."""
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1) + 2 * margin
    total_h = max(p.height for p in panels) + 2 * margin
    final = Image.new("RGB", (total_w, total_h), bg)

    x = margin
    for panel in panels:
        final.paste(panel, (x, margin))
        x += panel.width + gap
    return final
