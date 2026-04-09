"""Panel C: Model Comparison — pretrained scatter with coarse reference.

Scatter of pretrained models (supervised, self-supervised, vision-language)
with a dashed reference line for the best coarse-trained 8-class CLIP model.
"""

import sys
import sqlite3

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import DB_PATH

# ── Constants ────────────────────────────────────────────────────────────
COARSE_BAR_COLOR = "#08519c"   # dark blue (CLIP)

PRETRAINED_GROUPS = {
    "Supervised": [
        ("AlexNet",    "AlexNet",         "cnn"),
        ("VGG-16",     "VGG16",           "cnn"),
        ("ResNet-50",  "ResNet50",        "cnn"),
        ("ConvNeXt",   "ConvNeXt_Base",   "cnn"),
        ("ViT-B/16",   "ViTBase",         "vit"),
    ],
    "Self-supervised": [
        ("DINOv1",     "DINOv1_ResNet50", "cnn"),
        ("DINOv2",     "DINOv2_ViT_B14",  "vit"),
        ("DINOv3",     "DINOv3_ViT_L16",  "vit"),
    ],
    "Vision-language": [
        ("CLIP-B/32",  "CLIP_ViT_B32",    "vit"),
        ("CLIP-L/14",  "CLIP_ViT_L14",    "vit"),
    ],
}
GROUP_COLORS = {
    "Supervised":      "#4a8c6f",   # sage green
    "Self-supervised": "#6b5b95",   # muted lavender
    "Vision-language": "#c4377a",   # magenta-rose
}
ARCH_MARKERS = {"cnn": "p", "vit": "*"}


def _fetch_clip8_score():
    """Fetch CLIP 8-class score on THINGS (seed 1 only, matching pretrained evals)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT score, ci_low, ci_high FROM results
        WHERE neural_dataset = 'things-behavior' AND pca_labels_folder = 'pca_labels_clip'
          AND cfg_id = 8 AND compare_method = 'spearman' AND analysis = 'rsa'
          AND model_name = 'CustomCNN' AND epoch = 20 AND seed = 1
          AND reconstruct_from_pcs = 0
        ORDER BY score DESC LIMIT 1
    """, conn)
    conn.close()
    if df.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    r = df.iloc[0]
    return {
        "mean": r["score"],
        "ci_low": r["ci_low"] if pd.notna(r["ci_low"]) else r["score"],
        "ci_high": r["ci_high"] if pd.notna(r["ci_high"]) else r["score"],
        "label": "8 classes (CLIP repr.)",
    }


def _fetch_pretrained_data():
    """Fetch pretrained model scores for scatter."""
    conn = sqlite3.connect(DB_PATH)
    pretrained_df = pd.read_sql("""
        SELECT model_name, score, ci_low, ci_high
        FROM results
        WHERE neural_dataset = 'things-behavior'
          AND analysis = 'rsa'
          AND compare_method = 'spearman'
          AND cfg_id = 'pretrained'
    """, conn)
    conn.close()
    pretrained_df = (pretrained_df
                     .sort_values("ci_low", na_position="last")
                     .drop_duplicates("model_name", keep="first"))
    scores = pretrained_df.set_index("model_name")

    all_points = []
    for group_name, models in PRETRAINED_GROUPS.items():
        color = GROUP_COLORS[group_name]
        for display, db_name, arch in models:
            if db_name not in scores.index:
                continue
            row = scores.loc[db_name]
            all_points.append({
                "display": display, "score": row["score"],
                "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "color": color, "marker": ARCH_MARKERS[arch],
                "group": group_name,
            })
    all_points.sort(key=lambda p: p["score"], reverse=True)
    return all_points


def plot_comparison(ax, ref_ax=None):
    """Plot pretrained model scatter with coarse-grain 8-way reference line.

    If ref_ax is provided, syncs y-axis limits with it after plotting.
    """
    best_coarse = _fetch_clip8_score()
    all_points = _fetch_pretrained_data()

    # ── Layout: scatter groups evenly spaced ──
    group_positions = {
        "Supervised":      0.8,
        "Self-supervised": 3.8,
        "Vision-language": 6.1,
    }
    jitter_spread = 0.20

    # ── Dashed reference line for coarse-grain 8-way ──
    if not np.isnan(best_coarse["mean"]):
        x_ref_start = -0.3
        x_ref_end = list(group_positions.values())[-1] + 1.2
        ax.plot([x_ref_start, x_ref_end],
                [best_coarse["mean"], best_coarse["mean"]],
                color=COARSE_BAR_COLOR, linestyle=(0, (5, 3)),
                linewidth=1.0, alpha=0.55, zorder=1)
        ax.text(x_ref_start + 0.1, best_coarse["mean"] + 0.008,
                "8 classes (CLIP repr.)",
                ha="left", va="bottom", fontsize=9, color=COARSE_BAR_COLOR,
                fontstyle="italic")

    # ── Pre-group points and compute x positions ──
    pt_size_base = 150
    pt_size_star = 204
    grouped = {}
    for pt in all_points:
        grouped.setdefault(pt["group"], []).append(pt)

    for group_name, pts in grouped.items():
        gx = group_positions[group_name]
        n = len(pts)
        jitters = [gx] * n if n == 1 else list(np.linspace(gx - jitter_spread, gx + jitter_spread, n))
        for idx, pt in enumerate(pts):
            pt["x_plot"] = jitters[idx]

    # ── Scatter pretrained models ──
    for pt in all_points:
        ax.plot([pt["x_plot"], pt["x_plot"]], [pt["ci_low"], pt["ci_high"]],
                color=pt["color"], linewidth=1.8, alpha=0.55, zorder=4,
                solid_capstyle="round")
        sz = pt_size_star if pt["marker"] == "*" else pt_size_base
        ax.scatter(pt["x_plot"], pt["score"], marker=pt["marker"], c=pt["color"],
                   s=sz, edgecolors="white", linewidths=1.0, zorder=5)

    # ── Model name labels with downward repulsion ──
    fs_model = 8.5
    min_gap = 0.024
    group_label_x = {g: gx + 0.42 for g, gx in group_positions.items()}

    for group_name in PRETRAINED_GROUPS:
        group_pts = sorted(grouped.get(group_name, []),
                           key=lambda p: p["score"], reverse=True)
        label_ys = [pt["score"] for pt in group_pts]
        for _ in range(80):
            moved = False
            for i in range(len(label_ys) - 1):
                if label_ys[i] - label_ys[i + 1] < min_gap:
                    label_ys[i + 1] = label_ys[i] - min_gap
                    moved = True
            if not moved:
                break

        lx = group_label_x[group_name]
        for pt, ly in zip(group_pts, label_ys):
            if abs(ly - pt["score"]) > 0.005:
                ax.annotate("", xy=(pt["x_plot"] + 0.13, pt["score"]),
                            xytext=(lx - 0.03, ly),
                            arrowprops=dict(arrowstyle="-", color="#cccccc",
                                            lw=0.4, shrinkA=1, shrinkB=1),
                            zorder=2)
            ax.text(lx, ly, pt["display"],
                    ha="left", va="center", fontsize=fs_model, color="#444444",
                    zorder=10)

    # ── Axis formatting ──
    xlim_right = list(group_positions.values())[-1] + 2.3
    ax.set_xlim(-0.5, xlim_right)
    ax.set_ylabel("")
    ax.set_title("Model Comparison",
                 fontsize=11, fontweight="semibold", pad=8)

    scatter_xticks = [group_positions[g] for g in group_positions]
    scatter_xlabels = ["Supervised", "Self-\nsupervised", "Vision-\nlanguage"]
    ax.set_xticks(scatter_xticks)
    ax.set_xticklabels(scatter_xlabels, fontsize=11.4)

    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="major", direction="out")
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.tick_params(axis="x", which="major", direction="out")
    sns.despine(ax=ax, right=True, top=True, offset=4)

    # Sync y-axis with Panel B if provided (limits already set by plot_coarseness)
    if ref_ax is not None:
        ax.set_ylim(ref_ax.get_ylim())

    # ── Legend: CNN/ViT markers ──
    leg_handles = [
        Line2D([], [], marker="p", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.6,
               markersize=11, label="CNN"),
        Line2D([], [], marker="*", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=13, label="ViT"),
    ]
    leg = ax.legend(handles=leg_handles, fontsize=10, frameon=True,
                    loc="lower left", edgecolor="#dddddd", fancybox=False,
                    framealpha=0.95, handletextpad=0.3, borderpad=0.4,
                    labelspacing=0.3, ncol=1, columnspacing=0.5,
                    bbox_to_anchor=(0.0, 0.0))
    leg.get_frame().set_linewidth(0.4)
