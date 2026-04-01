"""Shared utilities for Figure 1 panels."""

import os

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "pc_scatter_1per_class.npz")

# ── Colors ──────────────────────────────────────────────────────────
PALETTE_2 = ["#1b9e77", "#d95f02"]
PALETTE_4 = ["#00A896", "#7B68EE", "#E8963E", "#D64045"]

# ── Inset configuration ──────────────────────────────────────────────
INSET_CLASSES = [
    (236, "n02738535", 12),   # Q0: armchair
    (249, "n02791124", 15),   # Q0: barber chair
    (613, "n04146614", 10),   # Q1: school bus
    (973, "n12833149",  3),   # Q2: African violet
    (8,   "n01641206", 15),   # Q3: wood frog
    (10,  "n01641577",  0),   # Q3: bullfrog
]


# ── Style setup ───────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib style for Figure 1."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── Label generation ─────────────────────────────────────────────────

def median_split_labels(pcs, n_way):
    """Assign labels by recursive median splits on PC axes."""
    n = len(pcs)
    labels = np.zeros(n, dtype=int)
    if n_way >= 2:
        med1 = np.median(pcs[:, 0])
        labels[pcs[:, 0] >= med1] = 1
    if n_way >= 4:
        new_labels = np.zeros(n, dtype=int)
        for half in [0, 1]:
            mask = labels == half
            med2 = np.median(pcs[mask, 1])
            new_labels[mask & (pcs[:, 1] < med2)] = half * 2
            new_labels[mask & (pcs[:, 1] >= med2)] = half * 2 + 1
        labels = new_labels
    return labels


# ── Image helpers ──────────────────────────────────────────────────────

def _get_inset_image(synset_id, imagenet_dir, size=52, image_index=5):
    """Load a representative image for a given ImageNet synset."""
    class_dir = os.path.join(imagenet_dir, synset_id)
    if not os.path.isdir(class_dir):
        return None
    imgs = sorted([f for f in os.listdir(class_dir)
                   if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    if not imgs:
        return None
    idx = min(image_index, len(imgs) - 1)
    path = os.path.join(class_dir, imgs[idx])
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def _repel_positions(points, min_dist=0.12, iterations=200, anchor_weight=0.01):
    """Push overlapping 2D positions apart iteratively with anchor pull."""
    pts = np.array(points, dtype=float)
    anchors = pts.copy()

    centroid = pts.mean(axis=0)
    for i in range(len(pts)):
        diff = pts[i] - centroid
        d = np.linalg.norm(diff)
        if d < min_dist * 0.5 and d > 1e-8:
            pts[i] = centroid + diff / d * min_dist * 0.6

    for _ in range(iterations):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < min_dist and d > 1e-8:
                    push = (min_dist - d) / 2.0 * diff / d * 1.3
                    pts[i] += push
                    pts[j] -= push
            pts[i] += anchor_weight * (anchors[i] - pts[i])
    return pts


def add_top_row_insets(ax, pcs, class_labels, label_array, colors, inset_classes,
                       imagenet_dir, zoom=0.32, thumb_size=52):
    """Overlay image insets on a scatter panel."""
    import matplotlib.patheffects as pe

    positions, thumbnails, border_colors = [], [], []

    for entry in inset_classes:
        cls_idx, synset_id = entry[0], entry[1]
        img_idx = entry[2] if len(entry) > 2 else 5
        idx = np.where(class_labels == cls_idx)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        thumb = _get_inset_image(synset_id, imagenet_dir, size=thumb_size,
                                 image_index=img_idx)
        if thumb is None:
            continue
        positions.append(pcs[idx])
        thumbnails.append(thumb)
        border_colors.append(colors[label_array[idx] % len(colors)])

    if not positions:
        return

    positions = np.array(positions)
    x_range = pcs[:, 0].max() - pcs[:, 0].min()
    y_range = pcs[:, 1].max() - pcs[:, 1].min()
    norm_pts = positions.copy()
    norm_pts[:, 0] = (norm_pts[:, 0] - pcs[:, 0].min()) / x_range
    norm_pts[:, 1] = (norm_pts[:, 1] - pcs[:, 1].min()) / y_range

    repelled = _repel_positions(norm_pts, min_dist=0.28, iterations=500,
                                anchor_weight=0.008)
    repelled = np.clip(repelled, -0.08, 1.08)
    repelled[:, 0] = repelled[:, 0] * x_range + pcs[:, 0].min()
    repelled[:, 1] = repelled[:, 1] * y_range + pcs[:, 1].min()

    for k in range(len(thumbnails)):
        im_box = OffsetImage(thumbnails[k], zoom=zoom)
        im_box.image.axes = ax
        c = border_colors[k]
        disp_x, disp_y = repelled[k]
        orig_x, orig_y = positions[k]

        dist = np.sqrt((disp_x - orig_x)**2 + (disp_y - orig_y)**2)
        if dist > 0.015:
            ax.annotate("", xy=(orig_x, orig_y),
                        xytext=(disp_x, disp_y),
                        arrowprops=dict(arrowstyle="-", color=c,
                                        lw=0.9, alpha=0.5,
                                        shrinkA=1, shrinkB=14),
                        zorder=5)

        ab = AnnotationBbox(
            im_box, (disp_x, disp_y),
            frameon=True, pad=0.10,
            bboxprops=dict(edgecolor=c, linewidth=2.8, facecolor="white",
                           alpha=0.97,
                           path_effects=[
                               pe.withStroke(linewidth=4.0, foreground="white"),
                           ]),
            zorder=6,
        )
        ax.add_artist(ab)
