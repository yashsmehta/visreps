"""Coarseness plot with log-scale x-axis for THINGS-behavior.

Single panel (no regions). Each architecture uses a distinct marker shape
and color. 1K baseline is a standalone point. Untrained baseline shown as
horizontal dashed line.

Usage:
    python manuscript/figures/plot_coarseness_log_things.py
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
]
N_ARCH = len(ARCHITECTURES)

# ── Style ─────────────────────────────────────────────────────────────────
ARCH_STYLE = {
    "alexnet": {"color": "#2166AC", "marker": "o"},
    "clip":    {"color": "#1B7837", "marker": "s"},
    "vit":     {"color": "#C51B7D", "marker": "^"},
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
    spread = np.linspace(-1, 1, n_arch)
    return 2 ** (spread[arch_idx] * 0.09)


def fetch_arch_data(folder):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def main():
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Untrained baseline
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], **UNTRAINED_STYLE, zorder=1)

    # 1K baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
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
        means, ci_lo, ci_hi = fetch_arch_data(folder)
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

    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.margins(x=0.04)

    ax.set_xlabel("Number of Classes", fontsize=10, labelpad=6)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=10, labelpad=6)
    ax.set_title("THINGS Behavior", fontsize=11, fontweight="semibold", pad=8)

    sns.despine(ax=ax, right=True, top=True, offset=5)

    # Legend
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

    ax.legend(handles=handles, loc="best", frameon=False, fontsize=9,
              handletextpad=0.3, columnspacing=1.0)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/coarseness_log_things.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
