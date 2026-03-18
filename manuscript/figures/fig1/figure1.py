"""Figure 1: Method Overview + Representation Analysis.

Composite figure (2 rows x 3 columns):
  Top row (A-C): Shared PCA scatter showing coarse-graining procedure
    A: 2-way coloring, B: 4-way coloring, C: 1000-way coloring
    All panels share the same (x, y) coordinates — only coloring changes.
  Bottom row (D-F): Learned representation PC scatter with image insets
    D: 2-way model (FC1), E: 4-way model (FC1), F: 1000-way pretrained (FC1)
    Each panel shows the model's OWN top-2 PCA projection of FC1 activations.

Usage (from project root):
    python manuscript/figures/fig1/figure1.py
    python manuscript/figures/fig1/figure1.py --recompute-top   # recompute shared PCA data
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import seaborn as sns

sys.path.insert(0, ".")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))


# ── Data paths ──────────────────────────────────────────────────────────
TOP_ROW_CACHE = os.path.join(SCRIPT_DIR, "pc_scatter_1per_class.npz")
DATA_2WAY = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                         "2pcs_compare", "data_2way_alexnet.npz")
DATA_4WAY = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                         "2pcs_compare", "data_4way_alexnet.npz")

OUTPUT = os.path.join(SCRIPT_DIR, "figure1.png")

# ── Colors ──────────────────────────────────────────────────────────────
# Top row: coarse-graining label colors
PALETTE_2 = ["#1b9e77", "#d95f02"]
PALETTE_4 = ["#2d6a4f", "#74c69d", "#e8963e", "#d64045"]  # dark green, light green, amber, red

# Bottom row: learned representation colors — slightly more saturated
# for better visibility on scatter with low alpha
REPR_COLORS_2 = ["#1b9e77", "#d95f02"]
REPR_COLORS_4 = ["#00997F", "#6B5CE7", "#E07D28", "#C93035"]

INSET_LAYER = "fc1"
N_INSETS = 3  # images per class

# ── Figure 1a inset configuration ──────────────────────────────────────
# 2 images per 4-way quadrant = 8 total. Each tuple: (class_idx, synset_id, image_index)
# image_index chosen for brightness and visual clarity.
# Q0 (man-made objects): armchair + barber chair
# 6 images: pairs for Q0 and Q3, single representative for Q1 and Q2
# Q0 (man-made objects): armchair + barber chair (pair)
# Q1 (vehicles): school bus (single representative, deep inside Q1)
# Q2 (plants): African violet (single representative, very deep inside Q2, bd=0.155)
# Q3 (vertebrates): wood frog + bullfrog (pair)
INSET_CLASSES = [
    (236, "n02738535", 12),   # Q0: armchair (bright, 600x600)
    (249, "n02791124", 15),   # Q0: barber chair (bright, 185x197)
    (613, "n04146614", 10),   # Q1: school bus (bright yellow, 500x374)
    (973, "n12833149",  3),   # Q2: African violet (bright, 500x366)
    (8,   "n01641206", 15),   # Q3: wood frog (bright, 500x278)
    (10,  "n01641577",  0),   # Q3: bullfrog (bright, 500x333)
]


def _get_inset_image(synset_id, imagenet_dir, size=52, image_index=5):
    """Load a representative image for a given ImageNet synset.

    Parameters
    ----------
    image_index : int
        Which image to pick from the sorted list (0=first, 5=sixth, etc.).
        Avoids edge-case first images that are sometimes atypical.
    """
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
        # Center crop to square before resizing
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def add_top_row_insets(ax, pcs, class_labels, label_array, colors, inset_classes,
                       imagenet_dir, zoom=0.32, thumb_size=52):
    """Overlay image insets on a top-row scatter panel.

    Parameters
    ----------
    ax : matplotlib Axes
    pcs : (N, 2) PCA coordinates
    class_labels : (N,) original 1000-way class index for each point
    label_array : (N,) labels used for coloring in THIS panel (2-way, 4-way, or 1000-way)
    colors : list of colors indexed by label_array values
    inset_classes : list of (class_idx, synset_id) tuples
    imagenet_dir : path to ImageNet train directory
    zoom : image zoom factor
    thumb_size : thumbnail pixel size
    """
    positions = []
    thumbnails = []
    border_colors = []

    for entry in inset_classes:
        cls_idx, synset_id = entry[0], entry[1]
        img_idx = entry[2] if len(entry) > 2 else 5
        # Find this class in the PCA array
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

    # Repel overlapping positions
    x_range = pcs[:, 0].max() - pcs[:, 0].min()
    y_range = pcs[:, 1].max() - pcs[:, 1].min()
    norm_pts = positions.copy()
    norm_pts[:, 0] = (norm_pts[:, 0] - pcs[:, 0].min()) / x_range
    norm_pts[:, 1] = (norm_pts[:, 1] - pcs[:, 1].min()) / y_range

    # Repel with stronger force to separate close pairs
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

        # Connector line from data point to inset
        dist = np.sqrt((disp_x - orig_x)**2 + (disp_y - orig_y)**2)
        if dist > 0.015:
            ax.annotate("", xy=(orig_x, orig_y),
                        xytext=(disp_x, disp_y),
                        arrowprops=dict(arrowstyle="-", color=c,
                                        lw=0.9, alpha=0.5,
                                        shrinkA=1, shrinkB=14),
                        zorder=5)

        import matplotlib.patheffects as pe
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


# ═══════════════════════════════════════════════════════════════════════
#  Top row: Shared PCA scatter (label space)
# ═══════════════════════════════════════════════════════════════════════

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


def plot_top_panel(ax, pcs, labels, n_classes, colors, title,
                   point_size=22, alpha=0.75, show_ylabel=True,
                   decision_lines=None, subtitle=None):
    """Draw one shared-PCA scatter panel (top row)."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))
    point_colors = np.array([colors[labels[i] % len(colors)] for i in order])

    ax.scatter(pcs[order, 0], pcs[order, 1],
               c=point_colors, s=point_size, alpha=alpha,
               edgecolors="white", linewidths=0.3,
               rasterized=True, zorder=2)

    ax.set_xlabel("PC 1", fontsize=10, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=10, labelpad=4)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="normal",
                     pad=8, linespacing=1.4)
        # Make the first line bold, second line smaller and gray
        t = ax.title
        t.set_fontweight("bold")
        # Use a two-part approach: bold title + smaller subtitle below
        ax.set_title("")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                fontsize=8.5, color="#666666", ha="center", va="bottom",
                fontstyle="italic")
    else:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.10
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=5, left=not show_ylabel)

    # Decision boundary lines
    if decision_lines is not None:
        split_kw = dict(color="#222222", linestyle="--", linewidth=1.3,
                        alpha=0.6, zorder=5)
        for line in decision_lines:
            if line["type"] == "vline":
                ax.axvline(line["pos"], **split_kw)
            elif line["type"] == "hline_segment":
                ax.plot(line["x"], [line["pos"], line["pos"]],
                        clip_on=True, **split_kw)


# ═══════════════════════════════════════════════════════════════════════
#  Bottom row: Learned representation PC scatter
# ═══════════════════════════════════════════════════════════════════════

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


_thumb_cache = {}


def _get_thumbnail(path, size=48):
    if path not in _thumb_cache:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            _thumb_cache[path] = np.array(img)
        except Exception:
            _thumb_cache[path] = None
    return _thumb_cache[path]


def _repel_positions(points, min_dist=0.12, iterations=200, anchor_weight=0.01):
    """Push overlapping 2D positions apart iteratively with anchor pull.

    First applies a radial expansion from the centroid to pre-separate tight
    clusters, then iteratively repels remaining overlaps.
    """
    pts = np.array(points, dtype=float)
    anchors = pts.copy()

    # Step 1: Radial pre-expansion from centroid
    centroid = pts.mean(axis=0)
    for i in range(len(pts)):
        diff = pts[i] - centroid
        d = np.linalg.norm(diff)
        if d < min_dist * 0.5 and d > 1e-8:
            # Push outward so each point is at least min_dist/2 from centroid
            pts[i] = centroid + diff / d * min_dist * 0.6

    # Step 2: Pairwise repulsion
    for _ in range(iterations):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < min_dist and d > 1e-8:
                    push = (min_dist - d) / 2.0 * diff / d * 1.3
                    pts[i] += push
                    pts[j] -= push
            # Gentle pull back toward anchor
            pts[i] += anchor_weight * (anchors[i] - pts[i])
    return pts


def plot_bottom_panel(ax, pcs, labels, n_classes, colors, title,
                      img_paths=None, inset_indices=None,
                      point_size=0.8, alpha=0.30, inset_zoom=0.38,
                      show_ylabel=True):
    """Draw one learned-representation PC scatter panel (bottom row)."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))
    pcs_s, labels_s = pcs[order], labels[order]

    for c in range(n_classes):
        mask = labels_s == c
        ax.scatter(pcs_s[mask, 0], pcs_s[mask, 1],
                   c=colors[c], s=point_size, alpha=alpha,
                   edgecolors="none", rasterized=True, zorder=2)

    ax.set_xlabel("PC 1", fontsize=10, labelpad=5, style="italic")
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=10, labelpad=5, style="italic")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#1a1a1a")
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.5, pad=3,
                   color="#999999")

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.08
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=6)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    # Image insets with repulsion to avoid overlap
    if inset_indices is not None and img_paths is not None:
        # Compute repelled display positions
        orig_pts = np.array([[pcs[ii, 0], pcs[ii, 1]] for ii in inset_indices])
        x_range = pcs[:, 0].max() - pcs[:, 0].min()
        y_range = pcs[:, 1].max() - pcs[:, 1].min()
        # Normalize to unit square for repulsion
        norm_pts = orig_pts.copy()
        norm_pts[:, 0] = (norm_pts[:, 0] - pcs[:, 0].min()) / x_range
        norm_pts[:, 1] = (norm_pts[:, 1] - pcs[:, 1].min()) / y_range
        repelled_norm = _repel_positions(norm_pts, min_dist=0.22, iterations=300,
                                        anchor_weight=0.012)
        # Clamp to stay within [−0.05, 1.05] in normalized space
        repelled_norm = np.clip(repelled_norm, -0.05, 1.05)
        # Convert back to data coordinates
        repelled = repelled_norm.copy()
        repelled[:, 0] = repelled_norm[:, 0] * x_range + pcs[:, 0].min()
        repelled[:, 1] = repelled_norm[:, 1] * y_range + pcs[:, 1].min()

        for k, ii in enumerate(inset_indices):
            thumb = _get_thumbnail(img_paths[ii], size=52)
            if thumb is None:
                continue
            im_box = OffsetImage(thumb, zoom=inset_zoom)
            im_box.image.axes = ax
            c = colors[int(labels[ii])]
            disp_x, disp_y = repelled[k]

            # Draw thin connector line from data point to inset
            dist = np.sqrt((disp_x - pcs[ii, 0])**2 + (disp_y - pcs[ii, 1])**2)
            if dist > 0.03:
                ax.annotate("", xy=(pcs[ii, 0], pcs[ii, 1]),
                            xytext=(disp_x, disp_y),
                            arrowprops=dict(arrowstyle="-", color=c,
                                            lw=0.8, alpha=0.45,
                                            shrinkA=2, shrinkB=18),
                            zorder=5)

            ab = AnnotationBbox(
                im_box, (disp_x, disp_y),
                frameon=True, pad=0.08,
                bboxprops=dict(edgecolor=c, linewidth=1.5, facecolor="white",
                               alpha=0.97),
                zorder=6,
            )
            ax.add_artist(ab)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-top", action="store_true",
                        help="Recompute shared PCA data (requires GPU + CLIP)")
    args = parser.parse_args()

    # ── Style ──
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

    # ── Load top row data (shared PCA, 1 image per class) ──
    if args.recompute_top or not os.path.exists(TOP_ROW_CACHE):
        # Recompute by running the extraction script
        from manuscript.figures.fig1.pc_scatter_explore import compute_and_cache
        top_data = compute_and_cache()
    else:
        print(f"Loading top-row cache: {TOP_ROW_CACHE}")
        raw = np.load(TOP_ROW_CACHE, allow_pickle=True)
        top_data = {k: raw[k] for k in raw.files}

    top_pcs = top_data["pcs"]
    top_class_labels = top_data["class_labels"]

    # Compute median-split labels for top row
    labels_2 = median_split_labels(top_pcs, 2)
    labels_4 = median_split_labels(top_pcs, 4)

    # Decision boundary positions
    med_pc1 = np.median(top_pcs[:, 0])
    med_pc2_left = np.median(top_pcs[top_pcs[:, 0] < med_pc1, 1])
    med_pc2_right = np.median(top_pcs[top_pcs[:, 0] >= med_pc1, 1])
    xlim_top = [top_pcs[:, 0].min() - (top_pcs[:, 0].max() - top_pcs[:, 0].min()) * 0.10,
                top_pcs[:, 0].max() + (top_pcs[:, 0].max() - top_pcs[:, 0].min()) * 0.10]

    # ── Load bottom row data ──
    has_2way = os.path.exists(DATA_2WAY)
    has_4way = os.path.exists(DATA_4WAY)

    if not has_4way:
        print(f"ERROR: 4-way data not found at {DATA_4WAY}")
        print("Run: python experiments/representation_analysis/2pcs_compare/run_analysis.py")
        return
    if not has_2way:
        print(f"WARNING: 2-way data not found at {DATA_2WAY}")
        print("Run: python experiments/representation_analysis/2pcs_compare/run_analysis.py "
              "--n_classes 2 --pca_labels_folder pca_labels_alexnet")
        print("Proceeding without 2-way panel...")

    # Load 4-way data (also contains the pretrained/1000-way model PCs)
    d4 = np.load(DATA_4WAY, allow_pickle=True)
    labels_4way = d4["pca_labels"]
    img_paths = d4["img_paths"]
    pcs_4way_trained = d4[f"{INSET_LAYER}_trained_pcs"].copy()
    pcs_pretrained = d4[f"{INSET_LAYER}_pretrained_pcs"].copy()

    # Align pretrained projection to 4-way trained via Procrustes
    pcs_pretrained_aligned, pcs_4way_trained = align_pc_projections(
        pcs_pretrained, pcs_4way_trained, labels_4way, 4)

    # Select inset images from 4-way trained panel (clear clusters)
    inset_idx_4 = select_inset_indices(pcs_4way_trained, labels_4way, 4)

    # Load 2-way data
    if has_2way:
        d2 = np.load(DATA_2WAY, allow_pickle=True)
        labels_2way = d2["pca_labels"]
        pcs_2way_trained = d2[f"{INSET_LAYER}_trained_pcs"].copy()
        pcs_2way_pretrained = d2[f"{INSET_LAYER}_pretrained_pcs"].copy()
        # Align 2-way pretrained to trained
        pcs_2way_pretrained, pcs_2way_trained = align_pc_projections(
            pcs_2way_pretrained, pcs_2way_trained, labels_2way, 2)
        inset_idx_2 = select_inset_indices(pcs_2way_trained, labels_2way, 2)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1a: Label space (2-class, 4-class, ..., 1000-class)
    # Layout: [2-class | 4-class | thin gap with progression | 1000-class]
    # ══════════════════════════════════════════════════════════════════════
    fig_a = plt.figure(figsize=(14, 4.8))
    # 2-class and 4-class close together; more space before 1000-class for progression
    gs_a = gridspec.GridSpec(1, 3, figure=fig_a,
                             width_ratios=[1, 1, 1.15],
                             wspace=0.15,
                             left=0.04, right=0.97, top=0.86, bottom=0.10)
    axes_a = [fig_a.add_subplot(gs_a[0, 0]),
              fig_a.add_subplot(gs_a[0, 1]),
              fig_a.add_subplot(gs_a[0, 2])]

    # Plot panels without titles — we'll draw a unified header instead
    plot_top_panel(axes_a[0], top_pcs, labels_2, 2, PALETTE_2, "",
                   show_ylabel=True,
                   decision_lines=[{"type": "vline", "pos": med_pc1}])

    plot_top_panel(axes_a[1], top_pcs, labels_4, 4, PALETTE_4, "",
                   show_ylabel=False,
                   decision_lines=[
                       {"type": "vline", "pos": med_pc1},
                       {"type": "hline_segment",
                        "x": [xlim_top[0], med_pc1], "pos": med_pc2_left},
                       {"type": "hline_segment",
                        "x": [med_pc1, xlim_top[1]], "pos": med_pc2_right},
                   ])

    rng_colors = np.random.RandomState(7)
    base_cmap = plt.cm.tab20
    colors_1k = []
    for i in range(1000):
        base = np.array(base_cmap(i % 20))
        jitter = rng_colors.uniform(-0.08, 0.08, 3)
        base[:3] = np.clip(base[:3] + jitter, 0, 1)
        colors_1k.append(tuple(base))
    rng_colors.shuffle(colors_1k)

    # Force highly distinct colors for the 6 inset classes in the 1000-way scheme
    # so their border colors are visually distinguishable from each other
    import matplotlib.colors as mcolors
    inset_1k_hex = [
        "#e41a1c",  # red        (Q0: armchair)
        "#377eb8",  # blue       (Q0: barber chair)
        "#4daf4a",  # green      (Q1: school bus)
        "#ff7f00",  # orange     (Q2: African violet)
        "#f781bf",  # pink       (Q3: wood frog)
        "#1b9e77",  # teal       (Q3: bullfrog)
    ]
    for k, entry in enumerate(INSET_CLASSES):
        cls_idx = entry[0]
        pos = np.where(top_class_labels == cls_idx)[0]
        if len(pos) > 0:
            colors_1k[pos[0]] = tuple(mcolors.to_rgba(inset_1k_hex[k]))

    plot_top_panel(axes_a[2], top_pcs, top_class_labels, 1000, colors_1k,
                   "", point_size=20, alpha=0.70, show_ylabel=False)

    # ── Add image insets to all three panels ──
    from dotenv import load_dotenv
    load_dotenv()
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR", "")
    if imagenet_dir and os.path.isdir(imagenet_dir):
        add_top_row_insets(axes_a[0], top_pcs, top_class_labels,
                           labels_2, PALETTE_2, INSET_CLASSES,
                           imagenet_dir, zoom=0.40, thumb_size=68)
        add_top_row_insets(axes_a[1], top_pcs, top_class_labels,
                           labels_4, PALETTE_4, INSET_CLASSES,
                           imagenet_dir, zoom=0.40, thumb_size=68)
        add_top_row_insets(axes_a[2], top_pcs, top_class_labels,
                           top_class_labels, colors_1k, INSET_CLASSES,
                           imagenet_dir, zoom=0.40, thumb_size=68)
    else:
        print(f"WARNING: ImageNet dir not found ({imagenet_dir}), skipping insets")

    # ── Unified header: progression 2 → 4 → 8 → 16 → 32 → 64 | break | 1000 ──
    pos0 = axes_a[0].get_position()
    pos1 = axes_a[1].get_position()
    pos2 = axes_a[2].get_position()
    header_y = pos0.y1 + 0.06   # main label y
    sub_y = pos0.y1 + 0.02      # subtitle y

    cx0 = (pos0.x0 + pos0.x1) / 2
    cx1 = (pos1.x0 + pos1.x1) / 2
    cx2 = (pos2.x0 + pos2.x1) / 2

    # Bold panel labels with "classes"
    fig_a.text(cx0, header_y, "2 classes", ha="center", va="bottom",
               fontsize=12, fontweight="bold", color="#1a1a1a",
               transform=fig_a.transFigure)
    fig_a.text(cx1, header_y, "4 classes", ha="center", va="bottom",
               fontsize=12, fontweight="bold", color="#1a1a1a",
               transform=fig_a.transFigure)
    fig_a.text(cx2, header_y, "1000 classes", ha="center", va="bottom",
               fontsize=12, fontweight="bold", color="#1a1a1a",
               transform=fig_a.transFigure)

    # Subtitles
    fig_a.text(cx0, sub_y, "median split on PC 1", ha="center", va="bottom",
               fontsize=8, color="#888888", fontstyle="italic",
               transform=fig_a.transFigure)
    fig_a.text(cx1, sub_y, "+ split on PC 2", ha="center", va="bottom",
               fontsize=8, color="#888888", fontstyle="italic",
               transform=fig_a.transFigure)
    fig_a.text(cx2, sub_y, "default ImageNet", ha="center", va="bottom",
               fontsize=8, color="#888888", fontstyle="italic",
               transform=fig_a.transFigure)

    # Intermediate progression: 8, 16, 32, 64 classes
    # Single line of text placed between "4 classes" and "1000 classes"
    gap_cx = (pos1.x1 + pos2.x0) / 2
    fig_a.text(gap_cx, header_y + 0.005,
               "8  ·  16  ·  32  ·  64 classes",
               ha="center", va="bottom",
               fontsize=9.5, fontweight="normal", color="#999999",
               transform=fig_a.transFigure)

    out_a = os.path.join(SCRIPT_DIR, "figure1a.png")
    fig_a.savefig(out_a, dpi=300, bbox_inches="tight", facecolor="white",
                  edgecolor="none")
    print(f"Saved -> {out_a}")

    out_a_svg = os.path.join(SCRIPT_DIR, "figure1a.svg")
    fig_a.savefig(out_a_svg, format="svg", bbox_inches="tight", facecolor="white",
                  edgecolor="none")
    print(f"Saved -> {out_a_svg}")
    plt.close(fig_a)

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1b: Learned representations (1000-way on top, 4-way on bottom)
    # ══════════════════════════════════════════════════════════════════════
    fig_b, axes_b = plt.subplots(2, 1, figsize=(6.5, 9.5))

    # 1000-way (pretrained) model on top — colored by 4-way labels
    plot_bottom_panel(
        axes_b[0], pcs_pretrained_aligned, labels_4way, 4, REPR_COLORS_4,
        "CNN trained on 1000 classes",
        img_paths=img_paths, inset_indices=inset_idx_4,
        point_size=1.0, alpha=0.35, inset_zoom=0.475,
        show_ylabel=True,
    )

    # 4-way model on bottom
    plot_bottom_panel(
        axes_b[1], pcs_4way_trained, labels_4way, 4, REPR_COLORS_4,
        "CNN trained on 4 derived classes",
        img_paths=img_paths, inset_indices=inset_idx_4,
        point_size=1.0, alpha=0.35, inset_zoom=0.475,
        show_ylabel=True,
    )

    fig_b.subplots_adjust(hspace=0.28, left=0.12, right=0.95, top=0.96, bottom=0.06)
    plt.tight_layout(h_pad=2.5)

    out_b = os.path.join(SCRIPT_DIR, "figure1b.png")
    fig_b.savefig(out_b, dpi=600, bbox_inches="tight", facecolor="white",
                  edgecolor="none")
    print(f"Saved -> {out_b}")
    plt.close(fig_b)


if __name__ == "__main__":
    main()
