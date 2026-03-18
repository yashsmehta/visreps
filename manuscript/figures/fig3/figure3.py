"""Figure 3: THINGS Behavioral Alignment.

Layout:
  Row 0 (top): [Schematic placeholder | Coarseness | Model Comparison]
  Row 1 (bottom): [4 PCA scatter panels] spanning full width

Panel A: Schematic of THINGS behavioral similarity task (placeholder)
Panel B: Alignment vs. Granularity (raw Spearman rho, log x-axis)
Panel C: Model comparison — coarse vs 1000-way bars + pretrained scatter
Panel D: PC scatter — Behavioral, CLIP 8-class, Pretrained AlexNet, Pretrained ViT

Usage:
    python manuscript/figures/fig3/figure3.py
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter, FixedLocator, NullLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory
from PIL import Image
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    DB_PATH,
    COARSE_CFGS, BREAK_1K_POS,
    UNTRAINED_LINE_STYLE, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
    format_coarseness_axes, draw_schematic_placeholder, draw_xaxis_break,
)
sys.path.insert(0, "manuscript/figures/fig3")
from plot_pc_scatter import (
    load_super_categories, l2_normalize, compute_pca,
    plot_scatter_panel as plot_pc_panel,
    SUPER_ORDER, SUPER_COLORS,
)

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig3"

# ── Figure 3 color scheme ──
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue
    "pixels":  {"color": "#c0a898", "marker": "v"},    # muted tan
}
BASELINE_1K_COLOR = "#d4822e"  # warm amber
COARSE_BAR_COLOR = "#08519c"   # dark blue (CLIP)

# ── Bar position for 1000-way (matches Fig 2 style) ──────────────────────
BAR_CENTER = 250
BAR_WIDTH_FRAC = 0.15

# ── Pretrained model comparison config ────────────────────────────────────
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
    "Vision-language": "#c4377a",   # magenta-rose (distinct from 1K orange)
}
ARCH_MARKERS = {"cnn": "p", "vit": "*"}


# ── Coarseness data fetching ─────────────────────────────────────────────

def fetch_things_arch_data(folder):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def _draw_bar_break(ax):
    """Draw // break marks between the coarse scatter region and the bar."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mid = np.exp((np.log(64) + np.log(BAR_CENTER / 1.16)) / 2)
    rect_hw = mid * 0.16
    rect = mpatches.FancyBboxPatch(
        (mid / 1.16, -0.05), width=rect_hw * 1.5, height=0.10,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none",
        transform=trans, clip_on=False, zorder=9)
    ax.add_patch(rect)
    for x_shift in [0.93, 1.07]:
        x_c = mid * x_shift
        ax.plot([x_c / 1.04, x_c * 1.04], [-0.028, 0.028],
                transform=trans, color="#555555", linewidth=0.7,
                clip_on=False, zorder=11)


def _make_tick_formatter(label_map):
    """Tolerance-based tick formatter for log-axis tick matching."""
    def _fmt(val, pos):
        for k, lbl in label_map.items():
            if abs(val - k) < k * 0.05:
                return lbl
        return ""
    return _fmt


def plot_coarseness_raw(ax):
    """Panel B: Raw Spearman rho coarseness for THINGS behavioral.

    Uses bar style for 1000-way baseline (matching Figure 2).
    """
    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_mean = bl["mean"]

    # Untrained baseline
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")

    # Collect y-values for axis range
    all_y_vals = []
    if not np.isnan(bl_mean):
        all_y_vals.append(bl_mean)
    if not np.isnan(un["mean"]):
        all_y_vals.append(un["mean"])

    # Coarse architectures
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_things_arch_data(folder)
        all_y_vals.extend([m for m in means if not np.isnan(m)])
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            ax.errorbar(cfg * jitter, means[i], yerr=[[e_lo], [e_hi]],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE,
                         markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                         capsize=1.5, capthick=0.5,
                         ecolor=style["color"], elinewidth=0.7, zorder=4)

    # Y-axis range (no forced zero)
    y_min = min(all_y_vals) if all_y_vals else 0
    y_max = max(all_y_vals) if all_y_vals else 1
    y_range = y_max - y_min
    y_bottom = y_min - y_range * 0.12

    # Untrained dashed line + label
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_LINE_STYLE, zorder=1)
        y_offset = y_range * 0.03
        ax.text(0.02, un["mean"] + y_offset, " Untrained",
                fontsize=6, fontstyle="italic", color="#AAAAAA",
                ha="left", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # 1000-way bar
    if not np.isnan(bl_mean):
        bl_err_lo = max(bl_mean - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
        bl_err_hi = max(bl["ci_high"] - bl_mean, 0) if not np.isnan(bl["ci_high"]) else 0
        ax.bar(BAR_CENTER, bl_mean - y_bottom, bottom=y_bottom,
               width=BAR_CENTER * BAR_WIDTH_FRAC,
               color=BASELINE_1K_COLOR, edgecolor="#c07830",
               linewidth=0.4, zorder=3)
        ax.errorbar(BAR_CENTER, bl_mean,
                    yerr=[[bl_err_lo], [bl_err_hi]],
                    fmt="none", ecolor="#555555", elinewidth=0.7,
                    capsize=2.2, capthick=0.6, zorder=5)

    # Axis formatting
    ax.set_xscale("log", base=2)
    all_ticks = COARSE_CFGS + [BAR_CENTER]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BAR_CENTER] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_make_tick_formatter(label_map)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6)
    ax.set_xlim(1.5, BAR_CENTER * 1.35)

    # Y-axis
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    # Small top margin
    cur_ylim = ax.get_ylim()
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + y_range * 0.03)

    ax.set_xlabel("ImageNet training classes", fontsize=8, labelpad=8)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=8.5, labelpad=3)
    sns.despine(ax=ax, right=True, top=True, offset=3)

    _draw_bar_break(ax)

    ax.set_title("Alignment vs. Granularity",
                 fontsize=9.5, fontweight="semibold", pad=8)


# ── Model comparison panel (simplified: 2 bars + pretrained scatter) ─────

def _fetch_clip8_score():
    """Fetch CLIP 8-class score on THINGS."""
    s = get_condition_summary("things-behavior", "N/A", "pca_labels_clip", 8,
                              "spearman", epoch=20, analysis="rsa")
    return {
        "mean": s["mean"],
        "ci_low": s["ci_low"],
        "ci_high": s["ci_high"],
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


def _draw_neurips_bar(ax, x, height, width, color, y_base=0.25,
                      edgecolor="black", linewidth=0.8, zorder=3):
    """Draw a bar with rounded top corners in NeurIPS style."""
    bar_bottom = y_base - 0.015
    bar_height = height - bar_bottom
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, bar_bottom), width, bar_height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, mutation_aspect=0.05, zorder=zorder,
    )
    ax.add_patch(rect)


def plot_comparison_panel(ax, ref_ax=None):
    """Panel C: Pretrained model scatter with coarse-grain 8-way reference line.

    If ref_ax is provided, syncs y-axis limits with it after plotting.
    """
    # ── Fetch coarse-grain 8-way reference score ──
    best_coarse = _fetch_clip8_score()
    all_points = _fetch_pretrained_data()

    # ── Layout: scatter groups evenly spaced across panel ──
    group_positions = {
        "Supervised":      2.0,
        "Self-supervised": 5.0,
        "Vision-language": 8.0,
    }
    jitter_spread = 0.30

    # ── Dashed reference line for coarse-grain 8-way ──
    if not np.isnan(best_coarse["mean"]):
        x_ref_start = -0.3
        x_ref_end = list(group_positions.values())[-1] + 1.2
        ax.plot([x_ref_start, x_ref_end],
                [best_coarse["mean"], best_coarse["mean"]],
                color=COARSE_BAR_COLOR, linestyle=(0, (5, 3)),
                linewidth=1.0, alpha=0.55, zorder=1)
        # Label just above the dashed line
        ax.text(x_ref_start + 0.1, best_coarse["mean"] + 0.008,
                "8 classes (CLIP repr.)",
                ha="left", va="bottom", fontsize=8.75, color=COARSE_BAR_COLOR,
                fontstyle="italic")

    # ── Draw pretrained scatter ──
    pt_size_base = 250
    pt_size_star = 340
    for pt in all_points:
        gx = group_positions[pt["group"]]
        group_pts = [p for p in all_points if p["group"] == pt["group"]]
        idx = group_pts.index(pt)
        n = len(group_pts)
        if n == 1:
            x_jit = gx
        else:
            x_jit = gx + np.linspace(-jitter_spread, jitter_spread, n)[idx]
        pt["x_plot"] = x_jit

        ax.plot([x_jit, x_jit], [pt["ci_low"], pt["ci_high"]],
                color=pt["color"], linewidth=1.8, alpha=0.55, zorder=4,
                solid_capstyle="round")
        sz = pt_size_star if pt["marker"] == "*" else pt_size_base
        ax.scatter(x_jit, pt["score"], marker=pt["marker"], c=pt["color"],
                   s=sz, edgecolors="white", linewidths=1.0, zorder=5)

    # ── Model name labels ──
    fs_model = 8.75
    x_offset = 0.38
    min_gap = 0.035
    for group_name in PRETRAINED_GROUPS:
        group_pts = sorted(
            [p for p in all_points if p["group"] == group_name],
            key=lambda p: p["score"], reverse=True)
        used_y = []
        for pt in group_pts:
            y = pt["score"]
            for uy in used_y:
                if abs(y - uy) < min_gap:
                    y = uy - min_gap
            used_y.append(y)
            ax.text(pt["x_plot"] + x_offset, y, pt["display"],
                    ha="left", va="center", fontsize=fs_model, color="#333333",
                    fontstyle="italic")

    # ── Axis formatting ──
    xlim_right = list(group_positions.values())[-1] + 2.4
    ax.set_xlim(-0.5, xlim_right)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9.5, labelpad=5)
    ax.set_title("Model Comparison",
                 fontsize=10.5, fontweight="semibold", pad=8)

    # X-ticks: only scatter group labels
    scatter_xticks = [group_positions[g] for g in group_positions]
    scatter_xlabels = ["Supervised", "Self-\nsupervised", "Vision-\nlanguage"]
    ax.set_xticks(scatter_xticks)
    ax.set_xticklabels(scatter_xlabels, fontsize=8.0)

    # Subtle horizontal grid
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Y-axis
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", which="major", direction="out", length=5,
                   width=1.2, labelsize=8.5)
    ax.tick_params(axis="y", which="minor", direction="out", length=3,
                   width=0.8)
    ax.tick_params(axis="x", which="major", length=4, width=1.0, direction="out")

    sns.despine(ax=ax, right=True, top=True, offset=5)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # ── Sync y-axis with Panel B if provided ──
    if ref_ax is not None:
        ref_ax.figure.canvas.draw()
        ylim = ref_ax.get_ylim()
        ax.set_ylim(ylim)

    # ── Legend: CNN/ViT markers only ──
    leg_handles = [
        Line2D([], [], marker="p", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.6,
               markersize=11, label="CNN"),
        Line2D([], [], marker="*", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=13, label="ViT"),
    ]
    leg = ax.legend(handles=leg_handles, fontsize=9, frameon=True,
                    loc="lower left", edgecolor="#dddddd", fancybox=False,
                    framealpha=0.95, handletextpad=0.3, borderpad=0.4,
                    labelspacing=0.3, ncol=1, columnspacing=0.5,
                    bbox_to_anchor=(0.0, 0.0))
    leg.get_frame().set_linewidth(0.4)


# ── Panel D — PCA scatter of THINGS concept representations ──────────────

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")

# Panel order and data config for PCA scatter
# (title, subtitle, data_key, needs_l2)
PC_PANELS = [
    ("Behavioral",  "(ground truth)", None),
    ("CNN",         "(8 classes (CLIP repr.))", "clip8"),
    ("AlexNet",     "(1K classes)",   "alexnet_pre"),
    ("ViT-B/16",    "(1K classes)",   "vit_pre"),
]


# ── Image inset config ────────────────────────────────────────────────────
# Triplet: cabbage (Food), cat (Animal), hubcap (Vehicle)
# Well-separated in behavioral/CLIP-8, close in AlexNet/ViT
THINGS_IMAGE_DIR = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/images/object_images")
INSET_CONCEPTS = ["asparagus", "engine", "gorilla"]
INSET_BORDER_COLORS = {
    "asparagus": SUPER_COLORS["Food"],      # orange
    "engine":    SUPER_COLORS["Vehicle"],   # green
    "gorilla":   SUPER_COLORS["Animal"],    # red
}


def _load_concept_names():
    """Load THINGS concept names from behavioral data."""
    behav_data = np.load(os.path.join(
        PROJECT_ROOT, "experiments/things_visualizations/data/things_viz_data.npz"),
        allow_pickle=True)
    return list(behav_data["concept_names"])


# Override default image variant for specific concepts (pick most recognizable at small size)
INSET_IMAGE_VARIANT = {
    "asparagus": "asparagus_04s.jpg",  # green bundle on white plate — high contrast
    "engine":    "engine_08s.jpg",     # silver V8 on black pedestal, white bg — clean silhouette
}

def _load_inset_image(concept, size=256):
    """Load and resize a THINGS concept image."""
    filename = INSET_IMAGE_VARIANT.get(concept, f"{concept}_01b.jpg")
    path = os.path.join(THINGS_IMAGE_DIR, concept, filename)
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


# Fallback offset angles if centroid-based directions are too close.
INSET_FALLBACK_ANGLES = {
    "asparagus": np.radians(210),   # lower-left
    "engine":    np.radians(330),   # lower-right
    "gorilla":   np.radians(90),    # top
}



def _shorten_line_to_circle(p1, p2, radius_frac=0.015):
    """Shorten a line segment so it stops at the edge of circles at both endpoints.

    radius_frac is the circle radius as a fraction of the axis range.
    Returns (x1', y1', x2', y2') — the shortened endpoints.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return p1[0], p1[1], p2[0], p2[1]
    ux, uy = dx / length, dy / length
    # Pull in from each end by the circle radius
    return (p1[0] + ux * radius_frac, p1[1] + uy * radius_frac,
            p2[0] - ux * radius_frac, p2[1] - uy * radius_frac)


def draw_image_insets(axes, all_pcs, concept_names):
    """Draw image insets for the triplet on each scatter panel.

    Places open circles at data positions, connects them with lines
    that stop at circle edges, and places images adjacent to circles.
    Uses a larger offset distance and connector lines from circle to image
    to keep the association clear even when points are clustered.
    """
    indices = [concept_names.index(c) for c in INSET_CONCEPTS]
    images = {c: _load_inset_image(c) for c in INSET_CONCEPTS}

    for ax, pcs in zip(axes, all_pcs):
        coords = {c: pcs[idx] for c, idx in zip(INSET_CONCEPTS, indices)}

        # Compute circle radius in data units (for line shortening)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        circle_radius = 0.025 * max(x_range, y_range)

        # Draw connecting lines between circles (stop at circle edges)
        for i in range(len(INSET_CONCEPTS)):
            c1 = INSET_CONCEPTS[i]
            c2 = INSET_CONCEPTS[(i + 1) % len(INSET_CONCEPTS)]
            x1s, y1s, x2s, y2s = _shorten_line_to_circle(
                coords[c1], coords[c2], circle_radius)
            ax.plot([x1s, x2s], [y1s, y2s],
                    color="#222222", linewidth=1.3, linestyle="-",
                    alpha=0.6, zorder=6, solid_capstyle="round")

        # Draw open circles at data positions
        for concept in INSET_CONCEPTS:
            x, y = coords[concept]
            ax.scatter(x, y, s=80, facecolors="none", edgecolors="black",
                       linewidths=1.5, zorder=7)

        # Place images: direction away from centroid, with minimum angular
        # separation enforced so images never bunch together.
        # Adaptive offset: close to circles but not overlapping.
        centroid = np.mean([coords[c] for c in INSET_CONCEPTS], axis=0)
        triplet_spread = max(
            np.ptp([coords[c][0] for c in INSET_CONCEPTS]) / x_range,
            np.ptp([coords[c][1] for c in INSET_CONCEPTS]) / y_range,
        )
        # Scale offset: 0.13 when clustered (AlexNet/ViT), up to 0.27 when spread
        # (Behavioral/CNN). Must clear circle radius + half tile size.
        offset_dist = (0.13 + 0.14 * min(triplet_spread / 0.4, 1.0)) * max(x_range, y_range)

        # Compute "away from centroid" angles
        angles = {}
        for concept in INSET_CONCEPTS:
            pt = coords[concept]
            away = pt - centroid
            norm = np.sqrt(away[0]**2 + away[1]**2)
            if norm > 1e-10:
                angles[concept] = np.arctan2(away[1], away[0])
            else:
                angles[concept] = INSET_FALLBACK_ANGLES[concept]

        # Enforce minimum angular separation (80°) between any pair
        min_sep = np.radians(80)
        concept_list = list(INSET_CONCEPTS)
        angle_list = [angles[c] for c in concept_list]
        for _ in range(20):
            changed = False
            for i in range(len(concept_list)):
                for j in range(i + 1, len(concept_list)):
                    diff = (angle_list[i] - angle_list[j] + np.pi) % (2 * np.pi) - np.pi
                    if abs(diff) < min_sep:
                        push = (min_sep - abs(diff)) * 0.3 * np.sign(diff)
                        angle_list[i] += push
                        angle_list[j] -= push
                        changed = True
            if not changed:
                break
        for i, c in enumerate(concept_list):
            angles[c] = angle_list[i]

        # Place images with minimal margin (scatter panels already have 8% data margins)
        margin_x = 0.02 * x_range
        margin_y = 0.02 * y_range
        # Minimum distance from circle center to image center
        min_dist_circle = 0.18 * max(x_range, y_range)
        # Minimum distance between tile centers (prevents tile overlap)
        min_dist_tiles = 0.12 * max(x_range, y_range)

        # First pass: compute initial positions
        tile_positions = {}
        x_lo = xlim[0] + margin_x
        x_hi = xlim[1] - margin_x
        y_lo = ylim[0] + margin_y
        y_hi = ylim[1] - margin_y
        for concept in INSET_CONCEPTS:
            pt = coords[concept]
            a = angles[concept]
            ox = pt[0] + np.cos(a) * offset_dist
            oy = pt[1] + np.sin(a) * offset_dist
            ox = np.clip(ox, x_lo, x_hi)
            oy = np.clip(oy, y_lo, y_hi)

            # After clamping, check if tile is too close to its circle.
            # If so, slide sideways (perpendicular) to maintain min distance.
            dx, dy = ox - pt[0], oy - pt[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist_circle:
                # Need to move perpendicular to reach min_dist_circle
                shortfall = np.sqrt(max(min_dist_circle**2 - dist**2, 0))
                # Perpendicular direction (rotate 90° from the radial direction)
                if dist > 1e-10:
                    perp_x, perp_y = -dy / dist, dx / dist
                else:
                    perp_x, perp_y = 1.0, 0.0
                # Try both perpendicular directions, pick the one within bounds
                for sign in [1, -1]:
                    nx = ox + sign * perp_x * shortfall
                    ny = oy + sign * perp_y * shortfall
                    if x_lo <= nx <= x_hi and y_lo <= ny <= y_hi:
                        ox, oy = nx, ny
                        break
                else:
                    # Both directions go out of bounds; just push radially
                    if dist > 1e-10:
                        scale = min_dist_circle / dist
                        ox = pt[0] + dx * scale
                        oy = pt[1] + dy * scale

            tile_positions[concept] = [ox, oy]

        # Second pass: push apart overlapping tiles (iterative repulsion)
        concept_list = list(INSET_CONCEPTS)
        for _ in range(30):
            moved = False
            for i in range(len(concept_list)):
                for j in range(i + 1, len(concept_list)):
                    ci, cj = concept_list[i], concept_list[j]
                    pi, pj = tile_positions[ci], tile_positions[cj]
                    dx = pi[0] - pj[0]
                    dy = pi[1] - pj[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < min_dist_tiles and dist > 1e-10:
                        push = (min_dist_tiles - dist) * 0.5
                        ux, uy = dx / dist, dy / dist
                        pi[0] += ux * push
                        pi[1] += uy * push
                        pj[0] -= ux * push
                        pj[1] -= uy * push
                        moved = True
            if not moved:
                break

        # Clamp again after repulsion
        for concept in INSET_CONCEPTS:
            pos = tile_positions[concept]
            pos[0] = np.clip(pos[0], xlim[0] + margin_x, xlim[1] - margin_x)
            pos[1] = np.clip(pos[1], ylim[0] + margin_y, ylim[1] - margin_y)

        # Shorten connector lines to 75% — move tiles closer to circles
        for concept in INSET_CONCEPTS:
            pt = coords[concept]
            ox, oy = tile_positions[concept]
            ox = pt[0] + (ox - pt[0]) * 0.75
            oy = pt[1] + (oy - pt[1]) * 0.75
            tile_positions[concept] = [ox, oy]

        # Draw connector lines and images
        for concept in INSET_CONCEPTS:
            pt = coords[concept]
            ox, oy = tile_positions[concept]

            ax.plot([pt[0], ox], [pt[1], oy],
                    color="#555555", linewidth=0.6, linestyle="-",
                    alpha=0.5, zorder=6)

            im = OffsetImage(images[concept], zoom=0.11)
            ab = AnnotationBbox(im, (ox, oy), frameon=True, pad=0.08,
                                bboxprops=dict(
                                    edgecolor="black",
                                    linewidth=0.8, facecolor="white",
                                ),
                                zorder=8)
            ax.add_artist(ab)


def load_pc_scatter_data():
    """Load the 4 representations for PCA scatter panels.

    Returns dict of name -> features (n_concepts, n_features).
    Model activations are L2-normalized before return.
    """
    behav_data = np.load(os.path.join(
        PROJECT_ROOT, "experiments/things_visualizations/data/things_viz_data.npz"),
        allow_pickle=True)
    activations = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig4/activations.npz"), allow_pickle=True)
    pretrained_alexnet = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig4/pretrained_alexnet_fc1.npz"))
    pretrained_vit = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig3/pretrained_vit_things.npz"))

    return {
        None:          behav_data["embeddings"],                   # (1854, 66)
        "clip8":       l2_normalize(activations["clip8_fc1"]),     # (1854, 4096)
        "alexnet_pre": l2_normalize(pretrained_alexnet["fc1"]),    # (1854, 4096)
        "vit_pre":     l2_normalize(pretrained_vit["block5"]),     # (1854, 151296)
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })

    fig = plt.figure(figsize=(15.5, 8.8))
    fig.patch.set_facecolor("white")

    # Unified 2×4 grid — columns align between rows
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           height_ratios=[1.05, 0.88],
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.38, wspace=0.28,
                           left=0.05, right=0.96,
                           top=0.95, bottom=0.07)

    # Panel A: cols 0–1 (schematic spans 2 columns)
    ax_schematic = fig.add_subplot(gs[0, 0:2])
    draw_schematic_placeholder(ax_schematic,
                               "THINGS\nBehavioral Similarity\n(schematic)")

    # Panel B: col 2
    ax_coarse = fig.add_subplot(gs[0, 2])
    plot_coarseness_raw(ax_coarse)

    # Panel B legend — architecture markers (bar is self-explanatory)
    legend_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=5.5, label=display)
        legend_handles.append(h)
    leg_c = ax_coarse.legend(
        handles=legend_handles, fontsize=7.5,
        frameon=True, fancybox=False, framealpha=0.92,
        edgecolor="#dddddd", borderpad=0.4,
        handletextpad=0.3, labelspacing=0.25,
        title="Latent repr. for\ncoarse labels",
        title_fontsize=7,
        loc="center left",
        bbox_to_anchor=(0.0, 0.45))
    leg_c._legend_box.align = "left"

    # Panel C: col 3
    ax_compare = fig.add_subplot(gs[0, 3])
    plot_comparison_panel(ax_compare, ref_ax=ax_coarse)

    # ── Bottom row: 4 PCA scatter panels (one per column) ──
    pc_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    print("Loading PCA scatter data...")
    reps = load_pc_scatter_data()
    n_concepts = list(reps.values())[0].shape[0]
    super_labels = load_super_categories(n_concepts)

    all_pcs = []
    for i, (ax, (title, subtitle, data_key)) in enumerate(zip(pc_axes, PC_PANELS)):
        feats = reps[data_key]
        pcs, _ = compute_pca(feats)
        all_pcs.append(pcs)
        plot_pc_panel(ax, pcs, super_labels, title, subtitle=subtitle,
                      point_size=12, alpha=0.62)
        if i > 0:
            ax.set_ylabel("")

    # Draw image insets on scatter panels
    concept_names = _load_concept_names()
    draw_image_insets(pc_axes, all_pcs, concept_names)

    # Super-category legend inside first scatter panel (upper-left)
    cat_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=SUPER_COLORS[name],
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=6, label=name)
        for name in SUPER_ORDER
    ]
    leg_cat = pc_axes[0].legend(
        handles=cat_handles, loc="upper left",
        ncol=2, fontsize=6.5, frameon=True,
        handletextpad=0.2, columnspacing=0.6, labelspacing=0.25,
        borderpad=0.3, edgecolor="#dddddd", fancybox=False,
        framealpha=0.90,
    )
    leg_cat.get_frame().set_linewidth(0.3)

    # ── Panel labels ──
    for ax, label, x_off in zip(
        [ax_schematic, ax_coarse, ax_compare, pc_axes[0]],
        ["A", "B", "C", "D"],
        [-0.08, -0.14, -0.06, -0.10]):
        ax.text(x_off, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
