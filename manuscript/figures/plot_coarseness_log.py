"""Coarseness plot with log-scale x-axis, all architectures overlaid.

Two panels (Early Visual Stream | Ventral Visual Stream). Each architecture
uses a distinct marker shape and color. 1K baseline is a standalone point.
Untrained baseline shown as horizontal dashed line. Points are jittered
horizontally to avoid overlap.

Usage:
    python manuscript/figures/plot_coarseness_log.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

# ── Data config ───────────────────────────────────────────────────────────
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
N_COARSE = len(COARSE_CFGS)

ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("vit",     "pca_labels_vit",     "ViT"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
N_ARCH = len(ARCHITECTURES)

REGIONS = [
    ("early visual stream",   "Early Visual Stream"),
    ("ventral visual stream", "Ventral Visual Stream"),
]

# ── Style ─────────────────────────────────────────────────────────────────
# Colorblind-friendly, saturated palette
ARCH_STYLE = {
    "alexnet": {"color": "#2166AC", "marker": "o"},   # Rich blue
    "clip":    {"color": "#1B7837", "marker": "s"},    # Forest green
    "dino":    {"color": "#E08214", "marker": "D"},    # Warm orange
    "vit":     {"color": "#C51B7D", "marker": "^"},    # Magenta-pink
    "pixels":  {"color": "#762A83", "marker": "v"},    # Purple
}
BASELINE_1K_COLOR = "#404040"
UNTRAINED_STYLE = {"color": "#AAAAAA", "linestyle": (0, (6, 3)), "linewidth": 1.4}

sns.set_theme(style="ticks", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})

OUTPUT_DIR = "manuscript/figures"
MARKER_SIZE = 6
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.5


def compute_jitter(arch_idx, n_arch):
    """Multiplicative jitter in log-space so points spread around each x tick."""
    spread = np.linspace(-1, 1, n_arch)
    return 2 ** (spread[arch_idx] * 0.09)


def fetch_arch_data(folder, region):
    """Fetch means + CIs for one architecture across all coarseness levels."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_panel(ax, region, region_label):
    """Draw one region panel with all architectures + baselines."""
    # Untrained baseline
    un = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_STYLE, zorder=1)

    # 1K baseline (standalone)
    bl = get_condition_summary("nsd", region, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    bl_err_lo = max(bl["mean"] - bl["ci_low"], 0) if not np.isnan(bl["ci_low"]) else 0
    bl_err_hi = max(bl["ci_high"] - bl["mean"], 0) if not np.isnan(bl["ci_high"]) else 0
    if not np.isnan(bl["mean"]):
        ax.errorbar(1000, bl["mean"], yerr=[[bl_err_lo], [bl_err_hi]],
                     fmt="D", color=BASELINE_1K_COLOR, markersize=6,
                     markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                     capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                     elinewidth=1.0, zorder=5)

    # Architecture points
    for arch_idx, (arch_key, folder, display) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_arch_data(folder, region)
        jitter = compute_jitter(arch_idx, N_ARCH)

        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
            e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
            x = cfg * jitter

            ax.errorbar(x, means[i], yerr=[[e_lo], [e_hi]],
                         fmt=style["marker"], color=style["color"],
                         markersize=MARKER_SIZE,
                         markeredgecolor=EDGE_COLOR,
                         markeredgewidth=EDGE_WIDTH,
                         capsize=2, capthick=0.7,
                         ecolor=style["color"], elinewidth=1.0,
                         zorder=4)

    # Axis formatting
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [1000]
    ax.set_xticks(all_x)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.tick_params(axis="y", which="major", direction="out", length=4, width=1.0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.6)

    # Subtle horizontal grid for cross-panel comparison
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)

    # Add x-margin so 1000 label isn't clipped
    ax.margins(x=0.04)

    ax.set_xlabel("Number of Classes", fontsize=10, labelpad=6)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=10, labelpad=6)
    ax.set_title(region_label, fontsize=11, fontweight="semibold", pad=8)

    sns.despine(ax=ax, right=True, top=True, offset=5)


def build_legend_handles():
    """Custom legend: one entry per architecture + 1K + untrained."""
    handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR,
                   markeredgewidth=EDGE_WIDTH,
                   markersize=MARKER_SIZE, label=display)
        handles.append(h)
    handles.append(Line2D([], [], marker="D", color="none",
                          markerfacecolor=BASELINE_1K_COLOR,
                          markeredgecolor=EDGE_COLOR,
                          markeredgewidth=EDGE_WIDTH,
                          markersize=6, label="1K (ImageNet)"))
    handles.append(Line2D([], [], linestyle=(0, (6, 3)), color="#AAAAAA",
                          linewidth=1.4, label="Untrained"))
    return handles


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, (region, label) in zip(axes, REGIONS):
        plot_panel(ax, region, label)

    handles = build_legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, handletextpad=0.3,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out = f"{OUTPUT_DIR}/coarseness_log.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
