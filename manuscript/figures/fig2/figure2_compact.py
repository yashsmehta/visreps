"""Figure 2 (compact): TVSD + NSD Neural Alignment — 2×2, no schematics.

Focused layout showing only coarse-grain results (2–64 classes) with the
1000-way trained model as a horizontal dashed reference line (no bootstrap).

  Row 0 (TVSD):  V1 | IT
  Row 1 (NSD):   Early visual stream | Ventral visual stream

Usage:
    python manuscript/figures/fig2/figure2_compact.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
)

OUTPUT_DIR = "manuscript/figures/fig2"

# ── Color scheme (matches original figure2.py) ─────────────────────────────
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
BASELINE_1K_COLOR = "#e8963e"    # warm amber


# ── Data fetching ────────────────────────────────────────────────────────

def fetch_nsd_arch_data(folder, region):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def fetch_nsd_baseline(region, epoch=20):
    s = get_condition_summary("nsd", region, "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"]


def _sem_summary(df):
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    return mean


def fetch_tvsd_arch_data(folder, region):
    means, sems = [], []
    for cfg in COARSE_CFGS:
        df = query_best_scores("tvsd", region, folder, cfg,
                               "spearman", epoch=20, analysis="rsa")
        if df.empty:
            means.append(np.nan)
            sems.append(0)
            continue
        seed_means = df.groupby("seed")["score"].mean()
        mean = seed_means.mean()
        sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
        means.append(mean)
        sems.append(sem)
    return np.array(means), np.array(sems)


def fetch_tvsd_baseline(region, epoch=20):
    df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                           "spearman", epoch=epoch, analysis="rsa")
    if df.empty:
        return np.nan
    return _sem_summary(df)


# ── Coarseness panel ─────────────────────────────────────────────────────

def plot_coarseness_compact(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Plot coarseness panel: coarse scatter + 1000-way dashed reference line."""

    # Fetch 1000-way baseline (mean only, no CI)
    if dataset == "nsd":
        bl_mean = fetch_nsd_baseline(region, epoch=20)
    else:
        bl_mean = fetch_tvsd_baseline(region, epoch=20)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    # ── 1) Architecture scatter points ──
    all_y_vals = [bl_mean]

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        if dataset == "nsd":
            means, ci_lo, ci_hi = fetch_nsd_arch_data(folder, region)
            errs_lo = np.array([max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
                                for i in range(len(means))])
            errs_hi = np.array([max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
                                for i in range(len(means))])
        else:
            means, sems = fetch_tvsd_arch_data(folder, region)
            errs_lo = 1.96 * sems
            errs_hi = 1.96 * sems

        all_y_vals.extend([m for m in means if not np.isnan(m)])
        jitter = compute_jitter(arch_idx, len(ARCHITECTURES))

        for i, cfg in enumerate(COARSE_CFGS):
            if np.isnan(means[i]):
                continue
            ax.errorbar(cfg * jitter, means[i],
                        yerr=[[errs_lo[i]], [errs_hi[i]]],
                        fmt=style["marker"], color=style["color"],
                        markersize=MARKER_SIZE,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                        capsize=1.5, capthick=0.5,
                        ecolor=style["color"], elinewidth=0.7, zorder=4)

    # ── 2) 1000-way dashed reference line + right-side label ──
    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=2)
    y_min = min(all_y_vals)
    y_max = max(all_y_vals)

    # ── 3) Axis formatting — only coarse classes on x-axis ──
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: str(int(val)) if int(round(val)) in set(COARSE_CFGS) else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6, labelsize=10)
    ax.set_xlim(1.5, 180)  # extra room right of 64 for label

    # Y-axis
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    # Y-range with small margins
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)

    # Right-side two-line label: "Trained on" / "1000 classes"
    y_offset = y_range * 0.015
    ax.text(180 * 0.95, bl_mean + y_offset, "Trained on\n1000 classes",
            fontsize=6, fontstyle="italic", color=BASELINE_1K_COLOR,
            ha="right", va="bottom", linespacing=1.1, zorder=10)

    if show_xlabel:
        ax.set_xlabel("ImageNet training classes", fontsize=9, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(8.5, 6.2))

    outer = gridspec.GridSpec(2, 2, figure=fig,
                              hspace=0.50, wspace=0.28,
                              height_ratios=[1, 1],
                              width_ratios=[1, 1],
                              left=0.11, right=0.96, top=0.86, bottom=0.10)

    axes = {}

    # ── Row 0: TVSD ──
    ax_v1 = fig.add_subplot(outer[0, 0])
    plot_coarseness_compact(ax_v1, "tvsd", "V1",
                            show_ylabel=True, show_xlabel=False)
    axes[(0, 0)] = ax_v1

    ax_it = fig.add_subplot(outer[0, 1])
    plot_coarseness_compact(ax_it, "tvsd", "IT",
                            show_ylabel=False, show_xlabel=False)
    axes[(0, 1)] = ax_it

    # ── Row 1: NSD ──
    ax_early = fig.add_subplot(outer[1, 0])
    plot_coarseness_compact(ax_early, "nsd", "early visual stream",
                            show_ylabel=True, show_xlabel=True)
    axes[(1, 0)] = ax_early

    ax_ventral = fig.add_subplot(outer[1, 1])
    plot_coarseness_compact(ax_ventral, "nsd", "ventral visual stream",
                            show_ylabel=False, show_xlabel=True)
    axes[(1, 1)] = ax_ventral

    # ── Column headers ──
    col_headers = {
        0: "Early Visual Cortex",
        1: "Higher Visual Cortex",
    }
    for col, header in col_headers.items():
        pos = axes[(0, col)].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        fig.text(x_center, y_top + 0.058, header,
                 fontsize=11.5, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom", family="sans-serif")

    # ── Per-panel subtitles (region names) ──
    row_subtitles = {
        (0, 0): ("V1", None),
        (0, 1): ("IT", None),
        (1, 0): ("Early visual stream", "(V1, V2, V3)"),
        (1, 1): ("Ventral visual stream", "(VO, PHC, and higher areas)"),
    }
    for ax_key, (line1, line2) in row_subtitles.items():
        ax = axes[ax_key]
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        if line2:
            fig.text(x_center, y_top + 0.025, line1,
                     fontsize=10, color="#888888",
                     ha="center", va="bottom", family="sans-serif")
            fig.text(x_center, y_top + 0.005, line2,
                     fontsize=8.125, color="#aaaaaa",
                     ha="center", va="bottom", family="sans-serif")
        else:
            fig.text(x_center, y_top + 0.012, line1,
                     fontsize=10, color="#888888",
                     ha="center", va="bottom", family="sans-serif")

    # ── Row labels (dataset + species) ──
    for row, (dataset, species) in [(0, ("TVSD", "Macaque")),
                                      (1, ("NSD", "Human"))]:
        pos = axes[(row, 0)].get_position()
        fy = (pos.y0 + pos.y1) / 2
        fig.text(0.018, fy, dataset, fontsize=10, fontweight="bold",
                 color="#2a2a2a", ha="center", va="center", rotation=90)
        fig.text(0.035, fy, species, fontsize=10,
                 color="#999999", ha="center", va="center", rotation=90)

    # ── Panel labels (a–d) ──
    row0_y = axes[(0, 0)].get_position().y1 + 0.068
    row1_y = axes[(1, 0)].get_position().y1 + 0.038

    label_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, key in enumerate(label_order):
        label = chr(ord("a") + i)
        ax = axes[key]
        pos = ax.get_position()
        x = pos.x0 - 0.02
        y = row0_y if key[0] == 0 else row1_y
        fig.text(x, y, label, fontsize=13, fontweight="bold",
                 va="bottom", ha="left", family="sans-serif")

    # ── Legend in panel a (TVSD V1) ──
    arch_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=5.5, label=display)
        arch_handles.append(h)
    leg = axes[(0, 0)].legend(
        handles=arch_handles, fontsize=7.5,
        frameon=True, fancybox=False, framealpha=0.92,
        edgecolor="#dddddd", borderpad=0.4,
        handletextpad=0.3, labelspacing=0.25,
        title="Coarse label source",
        title_fontsize=7,
        loc="center left",
        bbox_to_anchor=(0.0, 0.40),
    )
    leg._legend_box.align = "left"

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure2_compact.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
