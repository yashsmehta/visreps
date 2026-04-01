"""Supplementary Figure S7: Training Summary — Test accuracy vs granularity.

Left panel: scatter (broken x-axis, jittered markers, 4 PCA label sources, 2-64).
Right panel: narrow bar for 1000-way baseline (orange).

Usage:
    python manuscript/figures/supplementary/supp_s7_training_accuracy.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
import seaborn as sns

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    setup_style, COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
)

OUTPUT = "manuscript/figures/supplementary/supp_s7_training_accuracy.png"

SEED_LETTERS = ["a", "b", "c"]
FINAL_EPOCH = 20

# ── PCA label sources — same as S1 ──────────────────────────────────────
ARCHITECTURES = [
    ("alexnet", "/data/ymehta3/alexnet_pca", "AlexNet"),
    ("clip",    "/data/ymehta3/clip_pca",    "CLIP"),
    ("vit",     "/data/ymehta3/vit_pca",     "ViT"),
    ("dino",    "/data/ymehta3/dino_pca",    "DINO"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},
    "clip":    {"color": "#08519c", "marker": "s"},
    "vit":     {"color": "#c0392b", "marker": "^"},
    "dino":    {"color": "#1a8a7a", "marker": "p"},
}
BASELINE_1K_COLOR = "#e8963e"


def _compute_jitter(arch_idx, n_arch):
    """Minimal jitter — points are vertically distinct."""
    spread = np.linspace(-1, 1, n_arch)
    return 2 ** (spread[arch_idx] * 0.04)


def _load_final_accuracy(checkpoint_dir, cfg_id, seed_letter):
    if cfg_id == 1000:
        path = f"/data/ymehta3/default/cfg1000{seed_letter}/training_metrics.csv"
    else:
        path = f"{checkpoint_dir}/cfg{cfg_id}{seed_letter}/training_metrics.csv"
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    row = df[df["epoch"] == FINAL_EPOCH]
    if row.empty:
        return np.nan
    return row.iloc[0]["test_acc"]


def _load_accuracies(checkpoint_dir):
    """Load test accuracies for coarse conditions (2-64)."""
    data = {}
    for cfg in COARSE_CFGS:
        accs = [_load_final_accuracy(checkpoint_dir, cfg, sl) for sl in SEED_LETTERS]
        valid = [a for a in accs if not np.isnan(a)]
        if valid:
            mean = np.mean(valid)
            sem = np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0
            data[cfg] = (mean, sem)
    return data


def _load_baseline():
    """Load 1000-way test accuracy (mean, sem)."""
    accs = [_load_final_accuracy("", 1000, sl) for sl in SEED_LETTERS]
    valid = [a for a in accs if not np.isnan(a)]
    if valid:
        mean = np.mean(valid)
        sem = np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0
        return mean, sem
    return np.nan, 0


def _build_legend():
    """Legend with only the 4 PCA label sources."""
    return [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                   markerfacecolor=ARCH_STYLE[k]["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=6, label=d)
            for k, _, d in ARCHITECTURES]


def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[5, 0.8],
                           wspace=0.08,
                           left=0.10, right=0.95, top=0.92, bottom=0.14)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_scatter)

    all_y = []

    # ── Left panel: coarse scatter (2-64) ──
    for arch_idx, (arch_key, ckpt_dir, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        data = _load_accuracies(ckpt_dir)
        jitter = _compute_jitter(arch_idx, len(ARCHITECTURES))

        valid_x, valid_y = [], []
        for cfg in COARSE_CFGS:
            if cfg not in data:
                continue
            mean, sem = data[cfg]
            all_y.append(mean)
            x_pos = cfg * jitter
            valid_x.append(x_pos)
            valid_y.append(mean)
            ax_scatter.errorbar(x_pos, mean, yerr=sem,
                                fmt=style["marker"], color=style["color"],
                                markersize=MARKER_SIZE,
                                markeredgecolor=EDGE_COLOR,
                                markeredgewidth=EDGE_WIDTH,
                                capsize=1.5, capthick=0.5,
                                ecolor=style["color"], elinewidth=0.7,
                                zorder=4)

        # Light connecting line
        if len(valid_x) > 1:
            ax_scatter.plot(valid_x, valid_y, color=style["color"],
                            linewidth=0.8, alpha=0.3, zorder=3)

    # Scatter axis formatting
    ax_scatter.set_xscale("log", base=2)
    coarse_set = set(COARSE_CFGS)
    ax_scatter.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax_scatter.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: str(int(val)) if int(round(val)) in coarse_set else ""))
    ax_scatter.xaxis.set_minor_locator(NullLocator())
    ax_scatter.tick_params(axis="x", which="minor", bottom=False)
    ax_scatter.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax_scatter.set_xlim(1.5, 90)
    ax_scatter.set_xlabel("Granularity", fontsize=9, labelpad=5)
    ax_scatter.set_ylabel("Test accuracy (%)", fontsize=9, labelpad=4)

    ax_scatter.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax_scatter.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_scatter.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax_scatter.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax_scatter.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    sns.despine(ax=ax_scatter, right=True, top=True, offset=3)

    # Legend — top right, two-line title
    ax_scatter.legend(handles=_build_legend(), fontsize=7.5, frameon=True,
                      fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                      borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                      title="Coarse label\nsource", title_fontsize=7.5,
                      loc="upper right")

    # ── Right panel: 1000-way bar ──
    bl_mean, bl_sem = _load_baseline()
    if not np.isnan(bl_mean):
        all_y.append(bl_mean)
        ax_bar.bar(0, bl_mean, width=0.5, color=BASELINE_1K_COLOR,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax_bar.errorbar(0, bl_mean, yerr=bl_sem,
                        fmt="none", ecolor="#333333", elinewidth=0.8,
                        capsize=3, capthick=0.7, zorder=4)

    ax_bar.set_xticks([0])
    ax_bar.set_xticklabels(["1000"], fontsize=8)
    ax_bar.set_xlabel("", fontsize=9)
    ax_bar.set_xlim(-0.6, 0.6)
    ax_bar.tick_params(axis="y", labelleft=False)
    ax_bar.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax_bar.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_bar.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax_bar.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    sns.despine(ax=ax_bar, right=True, top=True, offset=3)

    # Shared y-limits
    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    ax_scatter.set_ylim(y_min - y_range * 0.10, y_max + y_range * 0.08)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
