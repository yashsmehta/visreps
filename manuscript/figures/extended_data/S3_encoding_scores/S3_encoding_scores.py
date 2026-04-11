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
import panel_raw as _fig3_panel_raw
from panel_raw import plot_raw as _fig3_plot_raw

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


# ── Monkey-patch fig3.panel_raw so plot_raw uses encoding-score fetchers ──
# plot_raw looks up ARCHITECTURES / ARCH_STYLE / fetch_* in its own module
# namespace at call time. Rebinding those names here makes plot_raw render
# encoding-score data with fig3's exact visual style (torn CI band included).
_fig3_panel_raw.ARCHITECTURES   = ARCHITECTURES
_fig3_panel_raw.ARCH_STYLE      = ARCH_STYLE
_fig3_panel_raw.fetch_arch_data = fetch_arch_data_enc
_fig3_panel_raw.fetch_baseline  = fetch_baseline_enc
_fig3_panel_raw.fetch_baseline_ci = fetch_baseline_ci_enc


def plot_raw_enc(ax, dataset, region, show_ylabel=True, show_xlabel=True,
                 show_untrained_label=False, tick_interval=None,
                 lollipop_ax=None):
    """Delegates to fig3.panel_raw.plot_raw (encoding-score fetchers patched above)."""
    _fig3_plot_raw(ax, dataset, region,
                   show_ylabel=show_ylabel, show_xlabel=show_xlabel,
                   show_untrained_label=show_untrained_label,
                   tick_interval=tick_interval, lollipop_ax=lollipop_ax)
    if show_ylabel:
        ax.set_ylabel("Encoding score (Pearson r)", fontsize=9, labelpad=4)


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

    # Match fig3's per-panel x-axis width (3.747"):
    # per_col = (right-left)*W / (n_cols + (n_cols-1)*wspace)
    # 3.747 = 0.87*W / 2.20  →  W ≈ 9.48
    fig = plt.figure(figsize=(9.48, 8.5))

    # 2 rows (TVSD | NSD) x 2 cols (early | higher) — no schematic column
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[1, 1],
                              width_ratios=[1, 1],
                              hspace=0.25, wspace=0.20,
                              left=0.10, right=0.97, top=0.88, bottom=0.08)

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
        ax_lol = fig.add_subplot(inner[0, 0], sharex=ax_raw)
        show_untrained = (orow == 1)
        plot_raw_enc(ax_raw, ds, region,
                     show_ylabel=ylabel, show_xlabel=xlabel,
                     show_untrained_label=show_untrained,
                     tick_interval=ytick, lollipop_ax=ax_lol)
        axes_scatter[(orow, ocol)] = ax_raw

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
    for row, title in [(0, "TVSD"), (1, "NSD")]:
        pos_l = axes_lollipop[(row, 0)].get_position()
        fig.text(0.015, (pos_l.y0 + pos_l.y1) / 2, title,
                 fontsize=13, fontweight="bold", color="#1a1a1a",
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
    # Place legend in panel b (IT), right side, just above the untrained
    # dashed grey line — mirrors fig3's bbox_to_anchor choice.
    axes_scatter[(0, 1)].legend(handles=handles, fontsize=7.5, frameon=True,
                                fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                                borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                                title="Coarse label source", title_fontsize=7.5,
                                loc="lower right", bbox_to_anchor=(1.0, 0.22))

    out = f"{OUTPUT_DIR}/S3_encoding_scores.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
