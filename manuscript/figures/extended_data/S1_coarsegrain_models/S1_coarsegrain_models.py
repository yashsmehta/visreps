"""Supplementary Figure S1: All PCA label sources across neural and behavioral benchmarks.

Same visual style as main Figure 3 panel_raw (raw Spearman rho, broken x-axis,
jittered scatter — NO lollipops, NO connecting lines).

Four PCA label sources per panel: AlexNet, CLIP, ViT, DINO.
Legend contains only the label sources (no 1000-way or untrained entries).

Generates two figures:
  S1A: supp_s1a_neural.png  — 2x2 grid (TVSD top | NSD bottom) x (Early | Higher)
  S1B: supp_s1b_behavioral.png — single THINGS panel

Usage:
    python manuscript/figures/extended_data/supp_s1_alternative_pca.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "manuscript/figures")
sys.path.insert(0, "manuscript/figures/fig3")
sys.path.insert(0, "plotters")

from fig_utils import (
    setup_style, COARSE_CFGS, BREAK_1K_POS, draw_xaxis_break,
    MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
)
from shared import fetch_baseline, fetch_baseline_ci, format_yaxis
from plotter_utils import get_condition_summary

# Import fig3's plotting module so we can reuse plot_raw (torn CI band, etc.)
# and monkey-patch its arch list to S1's 4 PCA sources.
import shared as _fig3_shared
import panel_raw as _fig3_panel_raw
from panel_raw import plot_raw as _fig3_plot_raw

OUTPUT_DIR = "manuscript/figures/extended_data/S1_coarsegrain_models"

# ── All four PCA label sources ──────────────────────────────────────────
# AlexNet + CLIP colors match Figure 3's shared.py exactly.
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("vit",     "pca_labels_vit",     "ViT"),
    ("dino",    "pca_labels_dino",    "DINO"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},   # medium blue  (same as fig3)
    "clip":    {"color": "#08519c", "marker": "s"},    # dark blue    (same as fig3)
    "vit":     {"color": "#c0392b", "marker": "^"},    # vivid crimson
    "dino":    {"color": "#1a8a7a", "marker": "p"},    # rich teal, pentagon
}
BASELINE_1K_COLOR = "#e8963e"


def _compute_jitter_wide(arch_idx, n_arch):
    """Wider jitter than fig_utils.compute_jitter for 4 overlapping sources."""
    spread = np.linspace(-1, 1, n_arch)
    return 2 ** (spread[arch_idx] * 0.14)  # 0.14 vs default 0.09


# ── Data fetching ────────────────────────────────────────────────────────

def _fetch_arch_data(dataset, folder, region):
    """Fetch means and asymmetric error bars for coarse conditions (2-64)."""
    means, errs_lo, errs_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary(dataset, region, folder, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        m = s["mean"]
        lo = max(m - s["ci_low"], 0) if not np.isnan(s["ci_low"]) else 0
        hi = max(s["ci_high"] - m, 0) if not np.isnan(s["ci_high"]) else 0
        means.append(m)
        errs_lo.append(lo)
        errs_hi.append(hi)
    return np.array(means), np.array(errs_lo), np.array(errs_hi)


def _fetch_things_data(folder):
    """Fetch coarseness scores for one PCA source on THINGS."""
    return _fetch_arch_data("things-behavior", folder, "N/A")


def _fetch_things_baseline(epoch=20):
    s = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


# ── Axis formatting ──────────────────────────────────────────────────────

def _format_broken_xaxis(ax, show_xlabel):
    """Log-2 x-axis with broken gap before 1000-way position."""
    ax.set_xscale("log", base=2)
    all_x = COARSE_CFGS + [BREAK_1K_POS]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BREAK_1K_POS] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_x))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: label_map.get(int(round(val)), "")))
    if show_xlabel:
        ax.set_xlabel("Granularity", fontsize=9, labelpad=5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, BREAK_1K_POS * 1.5)


# ── Panel plotting ───────────────────────────────────────────────────────

def plot_panel(ax, dataset, region, show_ylabel=True, show_xlabel=True,
               show_untrained_label=False, tick_interval=None):
    """Raw Spearman rho scatter for all 4 PCA sources — Figure 3 style.

    No connecting lines. No 1000-way / untrained in legend.
    """
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#999")
        return

    untrained_mean = fetch_baseline(dataset, region, epoch=0)
    all_y = [bl_mean]
    if not np.isnan(bl_ci_low):
        all_y.append(bl_ci_low)
    if not np.isnan(bl_ci_high):
        all_y.append(bl_ci_high)
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # Coarse conditions (2-64) — all 4 architectures, jittered
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = _fetch_arch_data(dataset, folder, region)
        for i, m in enumerate(means):
            if not np.isnan(m):
                all_y.extend([m - errs_lo[i], m + errs_hi[i]])
        jitter = _compute_jitter_wide(arch_idx, len(ARCHITECTURES))

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

    # 1000-way baseline: orange diamond at broken-axis position
    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    # Dashed reference line at 1000-way level
    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    # Untrained baseline
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.25, alpha=0.6, zorder=2)
        if show_untrained_label:
            ax.text(0.97, untrained_mean, "Untrained",
                    fontsize=8, fontstyle="italic", color="#999999",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), zorder=10)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min

    _format_broken_xaxis(ax, show_xlabel)
    draw_xaxis_break(ax)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    format_yaxis(ax, tick_interval=tick_interval)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def _build_legend():
    """Legend with only the 4 PCA label sources — no baselines."""
    return [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                   markerfacecolor=ARCH_STYLE[k]["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=6, label=d)
            for k, _, d in ARCHITECTURES]


# ── S1A: Neural (2x2 grid) ──────────────────────────────────────────────

def _precompute_fig3_ylim(dataset, region, tick_interval):
    """Dry-run fig3's plot_raw with its original 3 arches to capture its ylim."""
    fig_tmp, ax_tmp = plt.subplots(figsize=(3.75, 2.5))
    _fig3_plot_raw(ax_tmp, dataset, region,
                   show_ylabel=False, show_xlabel=False,
                   show_untrained_label=False,
                   tick_interval=tick_interval, lollipop_ax=None)
    ylim = ax_tmp.get_ylim()
    plt.close(fig_tmp)
    return ylim


def generate_s1a():
    """S1A: Neural data — 2 rows (TVSD | NSD) x 2 cols (Early | Higher).

    Matches main Figure 3: same panel_raw renderer (torn CI band, broken x-axis),
    same physical x-axis width per panel, and same y-axis limits as fig3.
    """
    # ── Pre-compute fig3 y-limits using its ORIGINAL 3-arch set ──
    panels = [
        # (row, col, dataset, region, show_ylabel, show_xlabel, show_untrained, tick_interval)
        # tick_interval=0.05 everywhere so the major-tick cadence matches
        # fig3's rendered panels (which auto-locate to 0.05 at fig3's size).
        (0, 0, "tvsd", "V1",                    True,  False, False, 0.05),
        (0, 1, "tvsd", "IT",                    False, False, False, 0.05),
        (1, 0, "nsd",  "early visual stream",   True,  True,  True,  0.05),
        (1, 1, "nsd",  "ventral visual stream", False, True,  False, 0.05),
    ]
    fig3_ylims = {
        (ds, region): _precompute_fig3_ylim(ds, region, ytick)
        for _, _, ds, region, _, _, _, ytick in panels
    }

    # ── Monkey-patch fig3 arches to S1's 4 PCA sources ──
    # Save originals so generate_s1b (which also uses fig3 utilities via
    # plotter_utils) isn't affected; restore in finally.
    _orig_shared_arches = _fig3_shared.ARCHITECTURES
    _orig_shared_style  = _fig3_shared.ARCH_STYLE
    _orig_pr_arches     = _fig3_panel_raw.ARCHITECTURES
    _orig_pr_style      = _fig3_panel_raw.ARCH_STYLE
    _fig3_shared.ARCHITECTURES    = ARCHITECTURES
    _fig3_shared.ARCH_STYLE       = ARCH_STYLE
    _fig3_panel_raw.ARCHITECTURES = ARCHITECTURES
    _fig3_panel_raw.ARCH_STYLE    = ARCH_STYLE
    # Also clear the lru_cache on fetch_arch_data so the dry-run's cached
    # results (from original arches) don't shadow the S1 arch fetches.
    try:
        _fig3_shared.fetch_arch_data.cache_clear()
    except AttributeError:
        pass

    # ── Figure geometry matching fig3's data-panel x-axis width (3.747") ──
    # fig3: figsize=(14,8.5), left=0.06, right=0.97, 3 cols, wspace=0.20
    #       per-col width = 14*(0.97-0.06)/(3+2*0.20) = 3.7471"
    # S1:   2 data cols, wspace=0.20, left=0.13, right=0.97
    #       figsize_w = 3.7471*(2+0.20)/(0.97-0.13) = 9.812"
    # Wider left margin so the rotated TVSD/NSD row labels don't collide
    # with the "RSA (Spearman ρ)" y-axis label of the left column.
    fig = plt.figure(figsize=(9.81, 6.8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.30, wspace=0.20,
                           left=0.13, right=0.97, top=0.87, bottom=0.10)

    axes = {}
    try:
        for row, col, ds, region, ylabel, xlabel, untrained_label, ytick in panels:
            ax = fig.add_subplot(gs[row, col])
            _fig3_plot_raw(ax, ds, region,
                           show_ylabel=ylabel, show_xlabel=xlabel,
                           show_untrained_label=untrained_label,
                           tick_interval=ytick, lollipop_ax=None)
            # Force y-limits to match fig3 exactly
            ax.set_ylim(*fig3_ylims[(ds, region)])
            axes[(row, col)] = ax
    finally:
        # Restore original fig3 arches
        _fig3_shared.ARCHITECTURES    = _orig_shared_arches
        _fig3_shared.ARCH_STYLE       = _orig_shared_style
        _fig3_panel_raw.ARCHITECTURES = _orig_pr_arches
        _fig3_panel_raw.ARCH_STYLE    = _orig_pr_style
        try:
            _fig3_shared.fetch_arch_data.cache_clear()
        except AttributeError:
            pass

    # Region titles
    region_titles = {
        (0, 0): "V1", (0, 1): "IT",
        (1, 0): "Early visual stream", (1, 1): "Ventral visual stream",
    }
    for key, label in region_titles.items():
        axes[key].set_title(label, fontsize=9.5, fontweight="medium",
                            color="#333333", pad=7)

    # Row headers — dataset labels
    # Place row headers to the LEFT of the y-axis label, clear of it.
    for row, (title, subtitle) in enumerate([
        ("TVSD", "Macaque electrophysiology"),
        ("NSD", "Human fMRI"),
    ]):
        pos = axes[(row, 0)].get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2, title,
                 fontsize=12, fontweight="bold", color="#1a1a1a",
                 ha="center", va="center", rotation=90)

    # Column headers — cortical level
    for col, label in [(0, "Early Visual Cortex"), (1, "Higher Visual Cortex")]:
        pos = axes[(0, col)].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.050, label,
                 fontsize=11, fontweight="bold", color="#333333",
                 ha="center", va="bottom")

    # Panel labels
    for idx, ((row, col), label) in enumerate(zip(
            [(0, 0), (0, 1), (1, 0), (1, 1)], "abcd")):
        pos = axes[(row, col)].get_position()
        fig.text(pos.x0 - 0.018, pos.y1 + 0.022, label,
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # Legend in panel d — right side, just above the untrained line
    axes[(1, 1)].legend(
        handles=_build_legend(), fontsize=7.5, frameon=True,
        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
        title="Coarse label source", title_fontsize=7.5,
        loc="right", bbox_to_anchor=(1.0, 0.30))

    out = f"{OUTPUT_DIR}/S1a_neural.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


# ── S1B: THINGS Behavioral (single panel) ───────────────────────────────

def generate_s1b():
    """S1B: THINGS behavioral — structurally identical to main Figure 4B.

    Reuses `fig4.panel_coarseness.plot_coarseness` directly (same torn CI
    band, same axes, same title, same legend styling) and monkey-patches
    its architecture list to S1's 4 PCA sources.
    """
    sys.path.insert(0, "manuscript/figures/fig4")
    import panel_coarseness as _fig4b  # noqa: WPS433

    # Save fig4B originals; restore in finally
    _orig_arches = _fig4b.ARCHITECTURES
    _orig_style  = _fig4b.ARCH_STYLE

    # Monkey-patch fig4B to use all 4 PCA sources. Reuse S1's ARCH_STYLE,
    # which already shares AlexNet/CLIP colors with fig4B and adds ViT/DINO.
    _fig4b.ARCHITECTURES = ARCHITECTURES
    _fig4b.ARCH_STYLE    = ARCH_STYLE

    # Match fig4B's physical panel size (figure4.py allocates ~one quarter
    # of a 14-inch-wide figure to panel B).
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
    try:
        _fig4b.plot_coarseness(ax)
    finally:
        _fig4b.ARCHITECTURES = _orig_arches
        _fig4b.ARCH_STYLE    = _orig_style

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/S1b_behavioral.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


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
    generate_s1a()
    generate_s1b()


if __name__ == "__main__":
    main()
