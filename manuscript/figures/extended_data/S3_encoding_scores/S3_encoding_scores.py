"""Figure 3 (encoding-score variant): Neural alignment — TVSD + NSD.

Same layout as figure3.py but:
  - No schematic column (data panels only)
  - Uses encoding score (Pearson r from voxelwise ridge regression)
    instead of RSA (Spearman rho)

2 rows (TVSD | NSD) x 2 cols (early cortex | higher cortex).
Each cell: horizontal lollipop strip (min classes to match 1K) above
           raw Pearson r scatter with broken x-axis.

Usage:
    python manuscript/figures/fig3/figure3_encoding.py
"""

import sys
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
    BREAK_1K_POS, draw_xaxis_break, setup_style,
)

sys.path.insert(0, "manuscript/figures/fig3")
from shared import ARCH_STYLE, BASELINE_1K_COLOR, format_yaxis
from panel_bits import LOLLIPOP_ARCHS, STEM_LW, MARKER_SZ, X_START, _draw_lollipop_break

OUTPUT_DIR = "manuscript/figures/extended_data/S3_encoding_scores"
ANALYSIS = "encoding_score"
COMPARE = "pearson"

# Pixels has ZERO encoding_score rows in results.db, so it's omitted here
# (unlike figure3.py which includes it for RSA).
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
]


# ── Encoding-score data fetchers (mirror shared.py but with analysis/compare overrides) ──

@lru_cache(maxsize=32)
def fetch_arch_data_enc(dataset, folder, region):
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary(dataset, region, folder, cfg,
                                  COMPARE, epoch=20, analysis=ANALYSIS)
        means.append(s["mean"]); ci_lo.append(s["ci_low"]); ci_hi.append(s["ci_high"])
    means = np.array(means)
    errs_lo = np.array([max(m - lo, 0) if not np.isnan(lo) else 0
                        for m, lo in zip(means, ci_lo)])
    errs_hi = np.array([max(hi - m, 0) if not np.isnan(hi) else 0
                        for m, hi in zip(means, ci_hi)])
    return means, errs_lo, errs_hi


@lru_cache(maxsize=16)
def fetch_baseline_enc(dataset, region, epoch=20):
    s = get_condition_summary(dataset, region, "imagenet1k", 1000,
                              COMPARE, epoch=epoch, analysis=ANALYSIS)
    return s["mean"]


@lru_cache(maxsize=16)
def fetch_baseline_ci_enc(dataset, region, epoch=20):
    s = get_condition_summary(dataset, region, "imagenet1k", 1000,
                              COMPARE, epoch=epoch, analysis=ANALYSIS)
    return s["mean"], s["ci_low"], s["ci_high"]


@lru_cache(maxsize=64)
def fetch_coarse_ci_enc(dataset, folder, region, cfg):
    s = get_condition_summary(dataset, region, folder, cfg,
                              COMPARE, epoch=20, analysis=ANALYSIS)
    return s["mean"], s["ci_low"], s["ci_high"]


# ── Panel renderers (replicate panel_raw/panel_bits with encoding fetchers) ──

def _format_broken_xaxis(ax, show_xlabel):
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: label_map.get(int(round(val)), "")))
        ax.set_xlabel("Granularity", fontsize=9, labelpad=4)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=10)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


def plot_raw_enc(ax, dataset, region, show_ylabel=True, show_xlabel=True,
                 show_untrained_label=False, tick_interval=None):
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci_enc(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    untrained_mean = fetch_baseline_enc(dataset, region, epoch=0)

    all_y = [bl_mean]
    if not np.isnan(bl_ci_low):  all_y.append(bl_ci_low)
    if not np.isnan(bl_ci_high): all_y.append(bl_ci_high)
    if not np.isnan(untrained_mean): all_y.append(untrained_mean)

    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = fetch_arch_data_enc(dataset, folder, region)
        for i, m in enumerate(means):
            if not np.isnan(m):
                all_y.extend([m - errs_lo[i], m + errs_hi[i]])
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

    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                   linewidth=1.25, alpha=0.6, zorder=2)
        if show_untrained_label:
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=8, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1e-3)

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax, tick_interval=tick_interval)

    if show_ylabel:
        ax.set_ylabel("Encoding score (Pearson r)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def _find_min_classes_enc(dataset, folder, region, bl_ci_low):
    for cfg in COARSE_CFGS:
        _, _, ci_high = fetch_coarse_ci_enc(dataset, folder, region, cfg)
        if not np.isnan(ci_high) and ci_high >= bl_ci_low:
            return cfg
    return np.nan


def plot_lollipop_enc(ax, dataset, region, show_ylabel=True):
    bl_mean, bl_ci_low, _ = fetch_baseline_ci_enc(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.axis("off")
        return

    ax.set_facecolor("#f5f5f5")
    y_positions = list(range(len(LOLLIPOP_ARCHS)))

    for i, (arch_key, folder, display) in enumerate(LOLLIPOP_ARCHS):
        style = ARCH_STYLE[arch_key]
        k_star = _find_min_classes_enc(dataset, folder, region, bl_ci_low)
        y = y_positions[i]

        if np.isnan(k_star):
            ax.plot([X_START, 64], [y, y], color=style["color"], linewidth=STEM_LW,
                    alpha=0.25, solid_capstyle="round", zorder=3)
            ax.text(64 * 1.15, y, "—", fontsize=7, color="#999",
                    va="center", ha="left")
            continue

        ax.plot([X_START, k_star], [y, y], color=style["color"], linewidth=STEM_LW,
                solid_capstyle="round", zorder=3)
        ax.plot(k_star, y, marker=style["marker"], color=style["color"],
                markersize=MARKER_SZ, markeredgecolor=EDGE_COLOR,
                markeredgewidth=EDGE_WIDTH, zorder=4)

    ax.axvline(BREAK_1K_POS, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=2)

    ax.set_ylim(-0.6, len(LOLLIPOP_ARCHS) - 0.4)
    ax.set_yticks(y_positions)
    if show_ylabel:
        ax.set_yticklabels([d for _, _, d in LOLLIPOP_ARCHS], fontsize=7.5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=3, width=0.7, pad=3)
    ax.tick_params(axis="x", labelbottom=False, length=0, bottom=False)
    sns.despine(ax=ax, bottom=True, right=True, top=True, left=False)
    ax.spines["left"].set_linewidth(0.7)
    _draw_lollipop_break(ax)


# ── Main ────────────────────────────────────────────────────────────────

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

    fig = plt.figure(figsize=(10, 8.5))

    # 2 rows (TVSD | NSD) x 2 cols (early | higher) — no schematic column
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[1, 1],
                              width_ratios=[1, 1],
                              hspace=0.28, wspace=0.22,
                              left=0.10, right=0.97, top=0.86, bottom=0.09)

    panel_defs = [
        # (row, col, dataset, region, show_ylabel, show_xlabel, tick_interval)
        (0, 0, "tvsd", "V1",                    True,  False, None),
        (0, 1, "tvsd", "IT",                    False, False, None),
        (1, 0, "nsd",  "early visual stream",   True,  True,  None),
        (1, 1, "nsd",  "ventral visual stream", False, True,  None),
    ]

    axes_scatter = {}
    axes_lollipop = {}

    for orow, ocol, ds, region, ylabel, xlabel, ytick in panel_defs:
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[orow, ocol],
            height_ratios=[0.14, 0.86], hspace=0.10)

        ax_raw = fig.add_subplot(inner[1, 0])
        show_untrained = (orow == 1)
        plot_raw_enc(ax_raw, ds, region,
                     show_ylabel=ylabel, show_xlabel=xlabel,
                     show_untrained_label=show_untrained,
                     tick_interval=ytick)
        axes_scatter[(orow, ocol)] = ax_raw

        ax_lol = fig.add_subplot(inner[0, 0], sharex=ax_raw)
        plot_lollipop_enc(ax_lol, ds, region, show_ylabel=True)
        axes_lollipop[(orow, ocol)] = ax_lol

    # Force-align lollipop plot areas to scatter plot areas
    for _ in range(2):
        fig.canvas.draw()
        for key in axes_lollipop:
            scat_pos = axes_scatter[key].get_position()
            lol_pos = axes_lollipop[key].get_position()
            axes_lollipop[key].set_position(
                [scat_pos.x0, lol_pos.y0, scat_pos.width, lol_pos.height])

    # Row headers (dataset name on the left)
    row_info = [(0, "TVSD", "Object images"), (1, "NSD", "Natural scenes")]
    for row, title, subtitle in row_info:
        pos_l = axes_lollipop[(row, 0)].get_position()
        pos_r = axes_lollipop[(row, 1)].get_position()
        y_top = max(pos_l.y1, pos_r.y1)
        fig.text(0.015, (pos_l.y0 + pos_l.y1) / 2, title,
                 fontsize=13, fontweight="bold", color="#1a1a1a",
                 ha="left", va="center", rotation=90)
        fig.text(0.040, (pos_l.y0 + pos_l.y1) / 2, subtitle,
                 fontsize=8, color="#777777", fontstyle="italic",
                 ha="left", va="center", rotation=90)

    # Column headers
    for col, label in [(0, "Early Visual Cortex"), (1, "Higher Visual Cortex")]:
        pos = axes_lollipop[(0, col)].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.045, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Region sub-labels
    region_labels = {
        (0, 0): "V1",       (0, 1): "IT",
        (1, 0): "Early visual stream", (1, 1): "Ventral visual stream",
    }
    for key, label in region_labels.items():
        pos = axes_lollipop[key].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, label,
                 fontsize=9, color="#666666", ha="center", va="bottom")

    # Panel labels (a–d)
    data_labels = [((0, 0), "a"), ((0, 1), "b"), ((1, 0), "c"), ((1, 1), "d")]
    for key, label in data_labels:
        pos = axes_lollipop[key].get_position()
        fig.text(pos.x0 - 0.022, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend
    handles = [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                      markerfacecolor=ARCH_STYLE[k]["color"],
                      markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                      markersize=6, label=d)
               for k, _, d in ARCHITECTURES]
    axes_scatter[(0, 0)].legend(handles=handles, fontsize=7.5, frameon=True,
                                fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                                borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                                title="Coarse label source", title_fontsize=7.5,
                                loc="upper right")

    out = f"{OUTPUT_DIR}/S3_encoding_scores.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
