"""Supplementary Figure S10: Additional PCA Source Models (ViT + DINOv3).

Shows normalized coarseness curves (% of 1000-way) for ViT- and
DINOv3-derived labels across all available datasets and regions.
These PCA source models are omitted from main Figures 3–4 for clarity.

Layout: 2x3 grid
  Row 0: NSD Early, NSD Ventral, TVSD V1
  Row 1: TVSD V4, TVSD IT, THINGS

Usage:
    python manuscript/figures/supplementary/supp_s10_dinov2.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, COARSE_CFGS, BASELINE_1K_COLOR, UNTRAINED_LINE_STYLE,
    normalize_to_baseline, format_normalized_coarseness_axes,
    MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
)
from plotters.plotter_utils import get_condition_summary, query_best_scores

OUTPUT = "manuscript/figures/supplementary/supp_s10_dinov2.png"

# Architecture styles
EXTRA_ARCHS = [
    ("vit",  "pca_labels_vit",  "ViT",    "#d62728", "^"),   # crimson
    ("dino", "pca_labels_dino", "DINOv3", "#17becf", "p"),   # teal cyan
]

# Panels: (neural_dataset, region, title)
PANELS = [
    ("nsd",             "early visual stream",   "NSD: Early Visual Stream"),
    ("nsd",             "ventral visual stream",  "NSD: Ventral Visual Stream"),
    ("tvsd",            "V1",                     "TVSD: V1"),
    ("tvsd",            "V4",                     "TVSD: V4"),
    ("tvsd",            "IT",                     "TVSD: IT"),
    ("things-behavior", "N/A",                    "THINGS: Behavior"),
]


def _sem_summary(df):
    """Mean and SEM across seeds (collapsing subjects first)."""
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
    return mean, sem


def fetch_coarse_data(dataset, region, folder):
    """Fetch coarse data for a given PCA folder: returns (means, ci_lo, ci_hi)."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        if dataset in ("nsd", "nsd_synthetic", "things-behavior"):
            s = get_condition_summary(dataset, region, folder, cfg,
                                      "spearman", epoch=20, analysis="rsa")
            means.append(s["mean"])
            ci_lo.append(s["ci_low"])
            ci_hi.append(s["ci_high"])
        elif dataset == "tvsd":
            df = query_best_scores(dataset, region, folder, cfg,
                                   "spearman", epoch=20, analysis="rsa")
            if df.empty:
                means.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan)
                continue
            m, sem = _sem_summary(df)
            means.append(m)
            ci_lo.append(m - 1.96 * sem)
            ci_hi.append(m + 1.96 * sem)
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def fetch_baseline(dataset, region, epoch=20):
    """Fetch 1000-way baseline."""
    if dataset in ("nsd", "nsd_synthetic", "things-behavior"):
        s = get_condition_summary(dataset, region, "imagenet1k", 1000,
                                  "spearman", epoch=epoch, analysis="rsa")
        return s["mean"], s["ci_low"], s["ci_high"]
    elif dataset == "tvsd":
        df = query_best_scores(dataset, region, "imagenet1k", 1000,
                               "spearman", epoch=epoch, analysis="rsa")
        if df.empty:
            return np.nan, np.nan, np.nan
        m, sem = _sem_summary(df)
        return m, m - 1.96 * sem, m + 1.96 * sem


def plot_panel(ax, dataset, region, title, show_ylabel=True, show_xlabel=True):
    """Plot one normalized coarseness panel for ViT + DINOv3."""
    bl_mean, bl_ci_lo, bl_ci_hi = fetch_baseline(dataset, region, epoch=20)
    un_mean, _, _ = fetch_baseline(dataset, region, epoch=0)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
        return

    scale = 100.0 / bl_mean

    # Untrained line
    if not np.isnan(un_mean):
        ax.axhline(un_mean * scale, **UNTRAINED_LINE_STYLE, zorder=1)

    # 1000-way diamond at 100%
    bl_err_lo = max(bl_mean - bl_ci_lo, 0) * scale if not np.isnan(bl_ci_lo) else 0
    bl_err_hi = max(bl_ci_hi - bl_mean, 0) * scale if not np.isnan(bl_ci_hi) else 0
    ax.errorbar(1000, 100.0, yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=2, capthick=0.7, ecolor=BASELINE_1K_COLOR,
                elinewidth=1.0, zorder=5)

    # Plot each architecture (ViT + DINOv3)
    for arch_idx, (_, folder, _, color, marker) in enumerate(EXTRA_ARCHS):
        means, ci_lo, ci_hi = fetch_coarse_data(dataset, region, folder)
        norm_means = means * scale
        errs_lo = np.array([max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
                            for i in range(len(means))]) * scale
        errs_hi = np.array([max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
                            for i in range(len(means))]) * scale

        jitter = compute_jitter(arch_idx, len(EXTRA_ARCHS))
        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, norm_means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=marker, color=color,
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=2, capthick=0.7,
                        ecolor=color, elinewidth=1.0, zorder=4)

    format_normalized_coarseness_axes(ax, "", show_ylabel=show_ylabel,
                                       show_xlabel=show_xlabel)
    ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)


def main():
    setup_style()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for idx, (dataset, region, title) in enumerate(PANELS):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        show_ylabel = (col == 0)
        show_xlabel = (row == 1)
        plot_panel(ax, dataset, region, title,
                   show_ylabel=show_ylabel, show_xlabel=show_xlabel)

    # Legend
    handles = []
    for _, _, display, color, marker in EXTRA_ARCHS:
        handles.append(Line2D([], [], marker=marker, color="none",
                              markerfacecolor=color,
                              markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                              markersize=MARKER_SIZE, label=display))
    handles.append(Line2D([], [], marker="D", color="none",
                          markerfacecolor=BASELINE_1K_COLOR,
                          markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                          markersize=MARKER_SIZE, label="1K (ImageNet)"))
    handles.append(Line2D([], [], **UNTRAINED_LINE_STYLE, label="Untrained"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    # Panel labels
    for idx, ax in enumerate(axes.flat):
        label = chr(ord("a") + idx)
        ax.text(-0.10, 1.12, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
