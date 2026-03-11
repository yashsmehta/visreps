"""Shared constants and helpers for manuscript figure scripts.

Centralizes style, data config, and common drawing functions used by
fig3/figure3.py, fig4/figure4.py, and fig5/figure5.py.
"""

import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from matplotlib.lines import Line2D
import seaborn as sns

DB_PATH = "results.db"

# ── Data config ───────────────────────────────────────────────────────────
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
GRAN_CFGS = COARSE_CFGS + [1000]

# ── Architecture definitions ──────────────────────────────────────────────
# (key, pca_labels_folder, display_name)
ARCHITECTURES_ALL = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("vit",     "pca_labels_vit",     "ViT"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCHITECTURES_NO_PIXELS = [a for a in ARCHITECTURES_ALL if a[0] != "pixels"]

# ── Style ─────────────────────────────────────────────────────────────────
ARCH_STYLE = {
    "alexnet": {"color": "#2166AC", "marker": "o"},
    "clip":    {"color": "#1B7837", "marker": "s"},
    "vit":     {"color": "#C51B7D", "marker": "^"},
    "pixels":  {"color": "#E08214", "marker": "v"},
}
BASELINE_1K_COLOR = "#404040"
UNTRAINED_LINE_STYLE = {"color": "#AAAAAA", "linestyle": (0, (6, 3)), "linewidth": 1.4}
MARKER_SIZE = 6
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.5

# Granularity color palette for per-layer plots — blue gradient (light → dark)
GRAN_COLORS = {
    2:    "#c6dbef",   # very light blue
    4:    "#9ecae1",   # light blue
    8:    "#6baed6",   # medium-light blue
    16:   "#4292c6",   # medium blue
    32:   "#2171b5",   # dark blue
    64:   "#084594",   # very dark blue
    1000: "#e6550d",   # orange-red (distinct from coarse gradient)
}
GRAN_MARKERS = {2: "o", 4: "s", 8: "^", 16: "D", 32: "v", 64: "p", 1000: "X"}


def setup_style():
    """Apply shared matplotlib/seaborn theme for manuscript figures."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def compute_jitter(arch_idx, n_arch):
    """Multiplicative jitter in log-space so points spread around each x tick."""
    spread = np.linspace(-1, 1, n_arch)
    return 2 ** (spread[arch_idx] * 0.09)


def format_coarseness_axes(ax, region_label, show_ylabel=True, show_xlabel=True):
    """Shared axis formatting for coarseness log-scale panels."""
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [1000]
    ax.set_xticks(all_x)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    if not show_xlabel:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", which="major", direction="out", length=4, width=1.0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.6)
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.margins(x=0.04)
    if show_xlabel:
        ax.set_xlabel("Number of Classes", fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    ax.set_title(region_label, fontsize=10, fontweight="semibold", pad=6)
    sns.despine(ax=ax, right=True, top=True, offset=4)


def build_coarseness_legend(architectures):
    """Build legend handles for coarseness panels."""
    handles = []
    for arch_key, _, display in architectures:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=MARKER_SIZE, label=display)
        handles.append(h)
    handles.append(Line2D([], [], marker="D", color="none",
                          markerfacecolor=BASELINE_1K_COLOR,
                          markeredgecolor=EDGE_COLOR,
                          markeredgewidth=EDGE_WIDTH,
                          markersize=MARKER_SIZE, label="1K (ImageNet)"))
    handles.append(Line2D([], [], linestyle=(0, (6, 3)),
                          color="#AAAAAA", linewidth=1.4,
                          label="Untrained"))
    return handles


def plot_reconstruction_panels(axes, neural_dataset, regions, coarse_config):
    """Draw dual reconstruction curves on a list of axes.

    Delegates to plot_reconstruction_panel which uses GRAN_COLORS-matched
    blue shades for the coarse curve.
    """
    for i, (ax, (region, label)) in enumerate(zip(axes, regions)):
        plot_reconstruction_panel(ax, neural_dataset, region,
                                  f"Reconstruction: {label}",
                                  coarse_config, show_ylabel=(i == 0))


def plot_reconstruction_panel(ax, neural_dataset, region, region_label,
                              coarse_config, show_ylabel=True):
    """Draw a single dual reconstruction curve panel.

    The coarse model curve uses the GRAN_COLORS blue shade matching its cfg_id.
    """
    from experiments.reconstruction_analysis.plot_utils import (
        query_reconstruction_curve, query_untrained_baseline,
        aggregate_curve,
    )

    FINE_COLOR = "#e6a200"       # golden amber — 1000-way
    UNTRAINED_COLOR = "#969696"  # grey

    fine_df = query_reconstruction_curve(neural_dataset, region)
    fine_agg = aggregate_curve(fine_df)
    cfg_id, checkpoint_dir = coarse_config[region]
    coarse_df = query_reconstruction_curve(
        neural_dataset, region, cfg_id=cfg_id, checkpoint_dir=checkpoint_dir,
    )
    coarse_agg = aggregate_curve(coarse_df)
    untrained = query_untrained_baseline(neural_dataset, region)

    # Look up the blue shade for this cfg_id
    coarse_color = GRAN_COLORS.get(cfg_id, "#2166ac")

    if fine_agg.empty and coarse_agg.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_title(region_label, fontsize=10, fontweight="semibold")
        return

    # Untrained baseline
    un_mean, un_lo, un_hi = untrained
    if not np.isnan(un_mean):
        if not np.isnan(un_lo):
            ax.axhspan(un_lo, un_hi, color=UNTRAINED_COLOR, alpha=0.08, zorder=0)
        ax.axhline(un_mean, color=UNTRAINED_COLOR, linestyle=":",
                    linewidth=1.3, label="Untrained", zorder=1)

    # 1000-way curve
    if not fine_agg.empty:
        k = fine_agg["pca_k"].values
        ax.fill_between(k, fine_agg["ci_low"].values, fine_agg["ci_high"].values,
                        color=FINE_COLOR, alpha=0.15, zorder=2)
        ax.plot(k, fine_agg["mean"].values, "-o", color=FINE_COLOR, markersize=3,
                linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
                label="1000-way (top-$k$ PCs)", zorder=3)

    # Coarse curve — uses the matching blue shade
    if not coarse_agg.empty:
        k_c = coarse_agg["pca_k"].values
        ax.fill_between(k_c, coarse_agg["ci_low"].values, coarse_agg["ci_high"].values,
                        color=coarse_color, alpha=0.15, zorder=2)
        ax.plot(k_c, coarse_agg["mean"].values, "-s", color=coarse_color, markersize=3,
                linewidth=1.5, markeredgecolor="white", markeredgewidth=0.5,
                label=f"{cfg_id}-way model (top-$k$ PCs)", zorder=3)

    # Axis formatting
    if not fine_agg.empty:
        k_all = fine_agg["pca_k"].values
    else:
        k_all = coarse_agg["pca_k"].values
    ax.set_xlabel("Number of PCs ($k$)", fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    ax.set_title(region_label, fontsize=10, fontweight="semibold", pad=6)
    ax.set_xticks(k_all)
    labeled = {1, 5, 10, 20, 30, 40, 50} | {int(k_all[0]), int(k_all[-1])}
    ax.set_xticklabels(
        [str(int(v)) if int(v) in labeled else "" for v in k_all], fontsize=7.5)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=2.5, width=0.6, direction="out")
    ax.yaxis.grid(True, linestyle="-", alpha=0.15, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=4)


# ── Per-layer helpers ─────────────────────────────────────────────────────

# All 14 layers (pre + post ReLU/BN)
LAYER_ORDER_FULL = [
    "conv1_pre", "conv1_post", "conv2_pre", "conv2_post",
    "conv3_pre", "conv3_post", "conv4_pre", "conv4_post",
    "conv5_pre", "conv5_post", "fc1_pre", "fc1_post",
    "fc2_pre", "fc2_post",
]
# Labels placed only at post-ReLU positions (every other tick)
LAYER_LABEL_POSITIONS = [i for i, l in enumerate(LAYER_ORDER_FULL) if l.endswith("_post")]
LAYER_LABELS_SHORT = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]

# Architecture folder → checkpoint dir mapping
_ARCH_CHECKPOINT_DIR = {
    "pca_labels_alexnet": "/data/ymehta3/alexnet_pca",
    "pca_labels_clip": "/data/ymehta3/clip_pca",
    "pca_labels_vit": "/data/ymehta3/vit_pca",
    "pca_labels_pixels": "/data/ymehta3/pixels_pca",
    "pca_labels_dino": "/data/ymehta3/dino_pca",
}
# Reverse mapping: checkpoint_dir → pca_labels_folder
_CHECKPOINT_TO_FOLDER = {v: k for k, v in _ARCH_CHECKPOINT_DIR.items()}


def get_layer_folder_from_coarse_config(coarse_config, region):
    """Derive the pca_labels_folder for per-layer plots from the coarse_config.

    This ensures the per-layer profile uses the same architecture as the
    reconstruction control panel.
    """
    _, checkpoint_dir = coarse_config[region]
    return _CHECKPOINT_TO_FOLDER.get(checkpoint_dir)

# Architecture folder → display name mapping
_ARCH_DISPLAY = {
    "pca_labels_alexnet": "Classes from AlexNet repr.",
    "pca_labels_clip": "Classes from CLIP repr.",
    "pca_labels_vit": "Classes from ViT repr.",
    "pca_labels_pixels": "Classes from Pixel repr.",
    "pca_labels_dino": "Classes from DINO repr.",
}


def find_best_architecture(neural_dataset, region):
    """Find the PCA architecture with the highest peak score for a region.

    Returns (pca_labels_folder, display_name) for the architecture whose
    best (cfg_id, seed) combination achieves the highest mean score.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT pca_labels_folder, cfg_id, AVG(score) as mean_score
        FROM results
        WHERE neural_dataset = ?
          AND region = ?
          AND compare_method = 'spearman'
          AND reconstruct_from_pcs = 0
          AND epoch = 20
          AND cfg_id != 1000
          AND analysis = 'rsa'
        GROUP BY pca_labels_folder, cfg_id
        ORDER BY mean_score DESC
        LIMIT 1
    """, conn, params=[neural_dataset, region])
    conn.close()

    if df.empty:
        return "pca_labels_alexnet", "AlexNet-PCA"

    folder = df.iloc[0]["pca_labels_folder"]
    display = _ARCH_DISPLAY.get(folder, folder)
    return folder, display


def fetch_layer_scores(neural_dataset, region, pca_labels_folder):
    """Fetch per-layer scores for all granularity levels + 1000-way + untrained.

    Returns dict: cfg_id -> {layer: mean_score}.  cfg_id=0 for untrained.
    Scores are averaged across all subjects and seeds.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT l.layer, l.score, r.seed, r.subject_idx, r.cfg_id, r.epoch
        FROM layer_selection_scores l
        JOIN results r ON l.run_id = r.run_id AND l.compare_method = r.compare_method
        WHERE r.compare_method = 'spearman'
          AND r.neural_dataset = ?
          AND r.region = ?
          AND r.reconstruct_from_pcs = 0
          AND (
            (r.pca_labels_folder = ? AND r.epoch = 20 AND r.cfg_id IN (2,4,8,16,32,64))
            OR (r.cfg_id = 1000 AND r.epoch = 20 AND r.pca_labels_folder = 'imagenet1k')
            OR (r.cfg_id = 1000 AND r.epoch = 0)
          )
    """, conn, params=[neural_dataset, region, pca_labels_folder])
    conn.close()

    if df.empty:
        return {}

    result = {}
    for (cfg_id, epoch), group in df.groupby(["cfg_id", "epoch"]):
        key = 0 if epoch == 0 else cfg_id
        layer_means = group.groupby("layer")["score"].mean()
        result[key] = {layer: layer_means[layer] for layer in layer_means.index}
    return result


def plot_per_layer_panel(ax, neural_dataset, region, pca_folder=None, title=None,
                         show_ylabel=True, show_xlabel=True):
    """Plot per-layer RSA scores averaged across all subjects/seeds.

    If pca_folder is None, auto-selects the best architecture for this region.
    Plots all 14 layers (pre + post ReLU) with labels at post-ReLU positions.
    """
    if pca_folder is None:
        pca_folder, arch_display = find_best_architecture(neural_dataset, region)
        print(f"  Per-layer [{region}]: auto-selected {arch_display} ({pca_folder})")
    else:
        arch_display = _ARCH_DISPLAY.get(pca_folder, pca_folder)

    all_scores = fetch_layer_scores(neural_dataset, region, pca_folder)

    for cfg_id in GRAN_CFGS:
        scores = all_scores.get(cfg_id, {})
        layers = [l for l in LAYER_ORDER_FULL if l in scores]
        if not layers:
            continue
        means = [scores[l] for l in layers]
        x = [LAYER_ORDER_FULL.index(l) for l in layers]
        ax.plot(x, means, marker=GRAN_MARKERS[cfg_id], color=GRAN_COLORS[cfg_id],
                markersize=4.5, linewidth=1.4, markeredgecolor="white",
                markeredgewidth=0.5, label=str(cfg_id), zorder=3, alpha=0.90)

    # Untrained
    un_scores = all_scores.get(0, {})
    if un_scores:
        layers = [l for l in LAYER_ORDER_FULL if l in un_scores]
        means = [un_scores[l] for l in layers]
        x = [LAYER_ORDER_FULL.index(l) for l in layers]
        ax.plot(x, means, color="#AAAAAA", linewidth=1.1, linestyle="--",
                label="Untrained", zorder=2, alpha=0.7)

    # X-axis: ticks at all 14 positions, labels only at post-ReLU (every other)
    ax.set_xticks(range(len(LAYER_ORDER_FULL)))
    if show_xlabel:
        tick_labels = [""] * len(LAYER_ORDER_FULL)
        for pos, label in zip(LAYER_LABEL_POSITIONS, LAYER_LABELS_SHORT):
            tick_labels[pos] = label
        ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=0)
        ax.set_xlabel("Layer", fontsize=9, labelpad=4)
    else:
        ax.set_xticklabels([])
    # Minor ticks at pre-ReLU positions for subtle visual grouping
    ax.tick_params(axis="x", which="major", length=3, width=0.6)
    ax.set_xlim(-0.5, len(LAYER_ORDER_FULL) - 0.5)

    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    if title:
        ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="out", length=4, width=1.0)
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.6)
    sns.despine(ax=ax, right=True, top=True, offset=4)

    return pca_folder, arch_display


def build_per_layer_legend():
    """Build legend handles for per-layer panels."""
    handles = []
    for cfg_id in GRAN_CFGS:
        h = Line2D([], [], marker=GRAN_MARKERS[cfg_id], color=GRAN_COLORS[cfg_id],
                   markersize=5, linewidth=1.4, markeredgecolor="white",
                   markeredgewidth=0.5, label=str(cfg_id))
        handles.append(h)
    handles.append(Line2D([], [], color="#AAAAAA", linewidth=1.2, linestyle="--",
                          label="Untrained"))
    return handles
