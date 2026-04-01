"""Supplementary Figure S7: NSD-Synthetic (Out-of-Distribution) Results.

Shows normalized coarseness curves (% of 1000-way) for NSD-Synthetic data
with AlexNet and CLIP PCA label architectures.

Layout: 1x2 (Early Visual Stream, Ventral Visual Stream)

Usage:
    python manuscript/figures/supplementary/supp_s7_nsd_synthetic.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, COARSE_CFGS, ARCH_STYLE, BASELINE_1K_COLOR, UNTRAINED_LINE_STYLE,
    normalize_to_baseline, format_normalized_coarseness_axes, build_coarseness_legend,
    MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
)
from plotters.plotter_utils import get_condition_summary

OUTPUT = "manuscript/figures/supplementary/extra/nsd_synthetic.png"

DATASET = "nsd_synthetic"

# Only AlexNet and CLIP available for NSD-Synthetic
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
]

REGIONS = [
    ("early visual stream",   "Early Visual Stream"),
    ("ventral visual stream",  "Ventral Visual Stream"),
]


def fetch_arch_data(folder, region):
    """Fetch coarse data for one architecture."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary(DATASET, region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def fetch_baseline(region, epoch=20):
    """Fetch 1000-way baseline for NSD-Synthetic."""
    s = get_condition_summary(DATASET, region, "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


def plot_panel(ax, region, region_label, show_ylabel=True):
    """Plot one normalized coarseness panel."""
    bl_mean, bl_ci_lo, bl_ci_hi = fetch_baseline(region, epoch=20)
    un_mean, _, _ = fetch_baseline(region, epoch=0)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.set_title(region_label, fontsize=10, fontweight="semibold", pad=6)
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

    # Plot each architecture
    n_arch = len(ARCHITECTURES)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, ci_lo, ci_hi = fetch_arch_data(folder, region)
        norm_means = means * scale
        errs_lo = np.array([max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
                            for i in range(len(means))]) * scale
        errs_hi = np.array([max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
                            for i in range(len(means))]) * scale

        jitter = compute_jitter(arch_idx, n_arch)

        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, norm_means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=2, capthick=0.7,
                        ecolor=style["color"], elinewidth=1.0, zorder=4)

    format_normalized_coarseness_axes(ax, region_label, show_ylabel=show_ylabel,
                                       show_xlabel=True)


def main():
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    for idx, (region, label) in enumerate(REGIONS):
        plot_panel(axes[idx], region, label, show_ylabel=(idx == 0))

    # Cap the early visual stream y-axis if untrained baseline stretches it
    for idx, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        if ymax > 150:
            # Find the untrained line value (dashed gray line)
            un_val = None
            for line in ax.get_lines():
                if line.get_linestyle() == "--" and len(set(line.get_ydata())) == 1:
                    un_val = line.get_ydata()[0]
            ax.set_ylim(ymin, 130)
            if un_val is not None and un_val > 130:
                ax.annotate(f"Untrained: {un_val:.0f}%",
                            xy=(0.02, 0.97), xycoords="axes fraction",
                            fontsize=7, color="#888888", fontstyle="italic",
                            va="top", ha="left")

    # Legend
    handles = build_coarseness_legend(ARCHITECTURES)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    # Panel labels
    for idx, ax in enumerate(axes):
        label = chr(ord("a") + idx)
        ax.text(-0.10, 1.12, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
