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
PALETTE_4 = ["#00A896", "#7B68EE", "#E8963E", "#D64045"]

# Bottom row: learned representation colors (same 4-color scheme for
# 4-way and 1000-way panels; 2-way uses its own 2-color scheme)
REPR_COLORS_2 = ["#1b9e77", "#d95f02"]
REPR_COLORS_4 = ["#00A896", "#7B68EE", "#E8963E", "#D64045"]

INSET_LAYER = "fc1"
N_INSETS = 3  # images per class


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
                   decision_lines=None):
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
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.10
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=5)

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


def plot_bottom_panel(ax, pcs, labels, n_classes, colors, title,
                      img_paths=None, inset_indices=None,
                      point_size=0.5, alpha=0.20, inset_zoom=0.35,
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

    ax.set_xlabel("PC 1", fontsize=9, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=9, labelpad=3)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color="#1a1a1a")
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.5, pad=2)

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
                bboxprops=dict(edgecolor=c, linewidth=2.0, facecolor="white"),
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

    # ── Figure layout: 2 rows x 3 columns ──
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.35, wspace=0.30,
        left=0.05, right=0.97, top=0.94, bottom=0.04,
    )

    # ── Top row: shared PCA scatter ──
    ax_top = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel A: 2-way
    plot_top_panel(ax_top[0], top_pcs, labels_2, 2, PALETTE_2, "2-way",
                   show_ylabel=True,
                   decision_lines=[{"type": "vline", "pos": med_pc1}])

    # Panel B: 4-way
    plot_top_panel(ax_top[1], top_pcs, labels_4, 4, PALETTE_4, "4-way",
                   show_ylabel=False,
                   decision_lines=[
                       {"type": "vline", "pos": med_pc1},
                       {"type": "hline_segment",
                        "x": [xlim_top[0], med_pc1], "pos": med_pc2_left},
                       {"type": "hline_segment",
                        "x": [med_pc1, xlim_top[1]], "pos": med_pc2_right},
                   ])

    # Panel C: 1000-way (unique color per class)
    rng_colors = np.random.RandomState(7)
    base_cmap = plt.cm.tab20
    colors_1k = []
    for i in range(1000):
        base = np.array(base_cmap(i % 20))
        jitter = rng_colors.uniform(-0.08, 0.08, 3)
        base[:3] = np.clip(base[:3] + jitter, 0, 1)
        colors_1k.append(tuple(base))
    rng_colors.shuffle(colors_1k)
    plot_top_panel(ax_top[2], top_pcs, top_class_labels, 1000, colors_1k,
                   "1000-way", point_size=20, alpha=0.70, show_ylabel=False)

    # ── Bottom row: learned representation PC scatter ──
    ax_bot = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Panel D: 2-way model
    if has_2way:
        plot_bottom_panel(
            ax_bot[0], pcs_2way_trained, labels_2way, 2, REPR_COLORS_2,
            "2-way model",
            img_paths=img_paths, inset_indices=inset_idx_2,
            point_size=0.8, alpha=0.25, inset_zoom=0.40,
            show_ylabel=True,
        )
    else:
        ax_bot[0].text(0.5, 0.5, "2-way data not available\n(run run_analysis.py)",
                       ha="center", va="center", transform=ax_bot[0].transAxes,
                       fontsize=9, color="#999", fontstyle="italic")
        ax_bot[0].set_title("2-way model", fontsize=11, fontweight="bold", pad=6)
        for spine in ax_bot[0].spines.values():
            spine.set_visible(False)
        ax_bot[0].set_xticks([])
        ax_bot[0].set_yticks([])

    # Panel E: 4-way model
    plot_bottom_panel(
        ax_bot[1], pcs_4way_trained, labels_4way, 4, REPR_COLORS_4,
        "4-way model",
        img_paths=img_paths, inset_indices=inset_idx_4,
        point_size=0.8, alpha=0.25, inset_zoom=0.40,
        show_ylabel=False,
    )

    # Panel F: 1000-way (pretrained) model — colored by 4-way labels
    # to show that the fine-grained model does NOT separate coarse categories
    plot_bottom_panel(
        ax_bot[2], pcs_pretrained_aligned, labels_4way, 4, REPR_COLORS_4,
        "1000-way model",
        img_paths=img_paths, inset_indices=inset_idx_4,
        point_size=0.8, alpha=0.25, inset_zoom=0.40,
        show_ylabel=False,
    )

    # ── Panel labels ──
    panel_labels = ["A", "B", "C", "D", "E", "F"]
    all_axes = ax_top + ax_bot
    for ax, label in zip(all_axes, panel_labels):
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif")

    # ── Row labels ──
    fig.text(0.01, 0.72, "Label space", fontsize=12, fontweight="semibold",
             rotation=90, va="center", ha="center", color="#555555")
    fig.text(0.01, 0.30, "Learned\nrepresentations", fontsize=12,
             fontweight="semibold", rotation=90, va="center", ha="center",
             color="#555555")

    # ── Save ──
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
