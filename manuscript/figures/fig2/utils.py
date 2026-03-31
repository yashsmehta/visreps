"""Shared utilities for Figure 2 panels."""

import os

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# ── Data paths ──────────────────────────────────────────────────────────
TOP_ROW_CACHE = os.path.join(SCRIPT_DIR, "pc_scatter_1per_class.npz")
DATA_2WAY = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                         "2pcs_compare", "data_2way_alexnet.npz")
DATA_4WAY = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                         "2pcs_compare", "data_4way_alexnet.npz")

# ── Colors ──────────────────────────────────────────────────────────────
PALETTE_2 = ["#1b9e77", "#d95f02"]
PALETTE_4 = ["#2d6a4f", "#74c69d", "#e8963e", "#d64045"]

REPR_COLORS_2 = ["#1b9e77", "#d95f02"]
REPR_COLORS_4 = ["#1B7A4F", "#50C888", "#E88A2A", "#D63540"]

INSET_LAYER = "fc1"
N_INSETS = 3

# ── Figure 2a inset configuration ──────────────────────────────────────
INSET_CLASSES = [
    (236, "n02738535", 12),   # Q0: armchair
    (249, "n02791124", 15),   # Q0: barber chair
    (613, "n04146614", 10),   # Q1: school bus
    (973, "n12833149",  3),   # Q2: African violet
    (8,   "n01641206", 15),   # Q3: wood frog
    (10,  "n01641577",  0),   # Q3: bullfrog
]


# ── Image helpers ──────────────────────────────────────────────────────

_thumb_cache = {}


def get_thumbnail(path, size=48):
    if path not in _thumb_cache:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            _thumb_cache[path] = np.array(img)
        except Exception:
            _thumb_cache[path] = None
    return _thumb_cache[path]


def get_inset_image(synset_id, imagenet_dir, size=52, image_index=5):
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


# ── Position repulsion ─────────────────────────────────────────────────

def repel_positions(points, min_dist=0.12, iterations=200, anchor_weight=0.01):
    """Push overlapping 2D positions apart iteratively with anchor pull."""
    pts = np.array(points, dtype=float)
    anchors = pts.copy()

    # Radial pre-expansion from centroid
    centroid = pts.mean(axis=0)
    for i in range(len(pts)):
        diff = pts[i] - centroid
        d = np.linalg.norm(diff)
        if d < min_dist * 0.5 and d > 1e-8:
            pts[i] = centroid + diff / d * min_dist * 0.6

    # Pairwise repulsion
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


# ── Inset overlay for top-row panels ──────────────────────────────────

def add_top_row_insets(ax, pcs, class_labels, label_array, colors, inset_classes,
                       imagenet_dir, zoom=0.32, thumb_size=52):
    """Overlay image insets on a top-row scatter panel."""
    import matplotlib.patheffects as pe

    positions = []
    thumbnails = []
    border_colors = []

    for entry in inset_classes:
        cls_idx, synset_id = entry[0], entry[1]
        img_idx = entry[2] if len(entry) > 2 else 5
        idx = np.where(class_labels == cls_idx)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        thumb = get_inset_image(synset_id, imagenet_dir, size=thumb_size,
                                image_index=img_idx)
        if thumb is None:
            continue
        positions.append(pcs[idx])
        thumbnails.append(thumb)
        border_colors.append(colors[label_array[idx] % len(colors)])

    if not positions:
        return

    positions = np.array(positions)

    # Repel overlapping positions
    x_range = pcs[:, 0].max() - pcs[:, 0].min()
    y_range = pcs[:, 1].max() - pcs[:, 1].min()
    norm_pts = positions.copy()
    norm_pts[:, 0] = (norm_pts[:, 0] - pcs[:, 0].min()) / x_range
    norm_pts[:, 1] = (norm_pts[:, 1] - pcs[:, 1].min()) / y_range

    repelled = repel_positions(norm_pts, min_dist=0.28, iterations=500,
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


# ── PCA alignment helpers ─────────────────────────────────────────────

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


def align_pc_projections(pretrained_pcs, trained_pcs, labels, n_classes):
    """Align pretrained PCA projection to trained via Procrustes on centroids."""
    pre_cent = np.array([pretrained_pcs[labels == c].mean(axis=0)
                         for c in range(n_classes)])
    tra_cent = np.array([trained_pcs[labels == c].mean(axis=0)
                         for c in range(n_classes)])
    pre_c = pre_cent - pre_cent.mean(axis=0)
    tra_c = tra_cent - tra_cent.mean(axis=0)
    U, _, Vt = np.linalg.svd(tra_c.T @ pre_c)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = (U @ Vt).T
    return pretrained_pcs @ R, trained_pcs


def discrete_align_pcs(pcs_to_align, pcs_reference, labels, n_classes):
    """Align via optimal sign flips to match quadrant arrangement."""
    centroids_ref = np.array([pcs_reference[labels == c].mean(axis=0)
                              for c in range(n_classes)])
    cr = centroids_ref - centroids_ref.mean(axis=0)
    cr /= np.maximum(cr.std(axis=0), 1e-8)

    best_flips, best_cost = (1, 1), np.inf
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            flipped = pcs_to_align * np.array([s1, s2])
            ca = np.array([flipped[labels == c].mean(axis=0) for c in range(n_classes)])
            ca = ca - ca.mean(axis=0)
            ca /= np.maximum(ca.std(axis=0), 1e-8)
            cost = np.sum((ca - cr) ** 2)
            if cost < best_cost:
                best_cost, best_flips = cost, (s1, s2)

    print(f"  Discrete alignment: flips = {best_flips}, cost = {best_cost:.4f}")
    return pcs_to_align * np.array(best_flips)


def select_inset_indices(pcs, labels, n_classes, n_per_class=N_INSETS):
    """Pick representative points spread across each class cluster."""
    indices = []
    for c in range(n_classes):
        mask = np.where(labels == c)[0]
        if len(mask) < n_per_class:
            indices.extend(mask.tolist())
            continue
        class_pcs = pcs[mask]
        centroid = class_pcs.mean(axis=0)
        dists = np.linalg.norm(class_pcs - centroid, axis=1)

        dist_threshold = np.percentile(dists, 85)
        inner_idx = np.where(dists < dist_threshold)[0]
        if len(inner_idx) < n_per_class:
            inner_idx = np.argsort(dists)[:max(n_per_class, len(mask) // 2)]

        offsets = class_pcs[inner_idx] - centroid
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        sector_edges = np.linspace(-np.pi, np.pi, n_per_class + 1)
        picks = []
        for s in range(n_per_class):
            in_sector = (angles >= sector_edges[s]) & (angles < sector_edges[s + 1])
            sector_pts = np.where(in_sector)[0]
            if len(sector_pts) == 0:
                continue
            sector_dists = dists[inner_idx[sector_pts]]
            target = max(0, int(len(sector_dists) * 0.4))
            pick = sector_pts[np.argsort(sector_dists)[target]]
            picks.append(inner_idx[pick])

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


# ── Mosaic helpers ────────────────────────────────────────────────────

def extract_class_indices(img_paths):
    """Map each image to a class index (0..N-1) via its synset directory."""
    synsets = np.array([str(p).split("/")[-2] for p in img_paths])
    unique_synsets = np.sort(np.unique(synsets))
    synset_to_idx = {s: i for i, s in enumerate(unique_synsets)}
    return np.array([synset_to_idx[s] for s in synsets]), unique_synsets


def compute_centroids(pcs, class_indices, n_classes):
    """Mean PC coordinates per class."""
    return np.array([pcs[class_indices == c].mean(axis=0) for c in range(n_classes)])


def closest_image_to_centroid(pcs, class_indices, img_paths, cls_idx, centroid):
    """Return path of the image closest to its class centroid in PC space."""
    mask = np.where(class_indices == cls_idx)[0]
    dists = np.linalg.norm(pcs[mask] - centroid, axis=1)
    return str(img_paths[mask[np.argmin(dists)]])


# ── Style setup ───────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib style for Figure 2."""
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
