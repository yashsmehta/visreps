"""Figure 2: Neural alignment across species — TVSD + NSD.

2 rows × 4 columns + schematic row on top:
  Columns grouped by dataset: TVSD (macaque) | NSD (human)
  Within each pair: alignment-per-bit (ρ/log₂K) | raw Spearman ρ
  Rows: early visual cortex (top) | higher visual cortex (bottom)

  Schematics with example stimuli + species icons span each dataset pair.
  Brain region insets (nilearn for human, SVG for macaque) on per-bit panels.

Usage:
    python manuscript/figures/fig2/figure2.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter,
)

sys.path.insert(0, "manuscript/figures/fig2")
from schematic_utils import draw_tvsd_schematic, draw_nsd_schematic, add_brain_inset

OUTPUT_DIR = "manuscript/figures/fig2"

# ── Color scheme ─────────────────────────────────────────────────────────
ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip",    "pca_labels_clip",    "CLIP"),
    ("pixels",  "pca_labels_pixels",  "Pixels"),
]
ARCH_STYLE = {
    "alexnet": {"color": "#6baed6", "marker": "o"},
    "clip":    {"color": "#08519c", "marker": "s"},
    "pixels":  {"color": "#c0a898", "marker": "v"},
}
BASELINE_1K_COLOR = "#e8963e"

# x-axis upper limit for raw panels (efficiency panels use a tighter range)
RAW_XLIM_RIGHT = 180


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
        means.append(seed_means.mean())
        sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
        sems.append(sem)
    return np.array(means), np.array(sems)


def fetch_tvsd_baseline(region, epoch=20):
    df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                           "spearman", epoch=epoch, analysis="rsa")
    if df.empty:
        return np.nan
    return df.groupby("seed")["score"].mean().mean()


# ── Shared helpers ──────────────────────────────────────────────────────

def _compute_error_bars(dataset, folder, region):
    """Compute means and asymmetric error bars for an architecture."""
    if dataset == "nsd":
        means, ci_lo, ci_hi = fetch_nsd_arch_data(folder, region)
        errs_lo = np.array([max(m - lo, 0) if not np.isnan(lo) else 0
                            for m, lo in zip(means, ci_lo)])
        errs_hi = np.array([max(hi - m, 0) if not np.isnan(hi) else 0
                            for m, hi in zip(means, ci_hi)])
    else:
        means, sems = fetch_tvsd_arch_data(folder, region)
        errs_lo = errs_hi = 1.96 * sems
    return means, errs_lo, errs_hi


def _fetch_baseline(dataset, region):
    if dataset == "nsd":
        return fetch_nsd_baseline(region)
    return fetch_tvsd_baseline(region)


def _format_xaxis(ax, show_xlabel, xlim_right=RAW_XLIM_RIGHT):
    ax.set_xscale("log", base=2)
    coarse_set = set(COARSE_CFGS)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    if show_xlabel:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: str(int(val)) if int(round(val)) in coarse_set else ""))
        ax.set_xlabel("Training classes", fontsize=9, labelpad=4)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.7, labelsize=8)
    ax.set_xlim(1.5, xlim_right)


def _format_yaxis(ax, fmt_str=".2f"):
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:{fmt_str}}".rstrip("0").rstrip(".")))


# ── Panel renderers ─────────────────────────────────────────────────────

def plot_raw(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """Raw Spearman ρ scatter (all architectures) + 1000-way dashed line."""
    bl_mean = _fetch_baseline(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    all_y = [bl_mean]
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = _compute_error_bars(dataset, folder, region)
        all_y.extend(m for m in means if not np.isnan(m))
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

    ax.axhline(bl_mean, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=2)

    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min

    _format_xaxis(ax, show_xlabel)
    ax.set_ylim(y_min - y_range * 0.12, y_max + y_range * 0.10)
    _format_yaxis(ax)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)


def plot_efficiency(ax, dataset, region, show_ylabel=True, show_xlabel=True):
    """CLIP-only alignment-per-bit with connected line + 1000-way reference."""
    bl_mean = _fetch_baseline(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    bl_eff = bl_mean / np.log2(1000)
    bits = np.array([np.log2(cfg) for cfg in COARSE_CFGS])

    means, errs_lo, errs_hi = _compute_error_bars(dataset, "pca_labels_clip", region)
    eff_means = means / bits
    errs_lo = errs_lo / bits
    errs_hi = errs_hi / bits

    valid = ~np.isnan(eff_means)
    x_vals = np.array(COARSE_CFGS)[valid]
    y_vals = eff_means[valid]

    clip_color = ARCH_STYLE["clip"]["color"]
    ax.plot(x_vals, y_vals, "-", color=clip_color, linewidth=1.4, zorder=3)
    ax.errorbar(x_vals, y_vals, yerr=[errs_lo[valid], errs_hi[valid]],
                fmt="s", color=clip_color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5,
                capsize=1.5, capthick=0.5,
                ecolor=clip_color, elinewidth=0.6, zorder=4)

    ax.axhline(bl_eff, color=BASELINE_1K_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=2)

    all_y = np.concatenate([y_vals, [bl_eff]])

    # Left-aligned "Trained on / 1000 classes" label
    y_range_eff = all_y.max() - all_y.min()
    ax.text(0.03, bl_eff + y_range_eff * 0.04,
            "Trained on\n1000 classes",
            fontsize=6.5, fontstyle="italic", color=BASELINE_1K_COLOR,
            ha="left", va="bottom", linespacing=1.1,
            transform=ax.get_yaxis_transform(), zorder=10)
    y_min, y_max = all_y.min(), all_y.max()
    y_range = y_max - y_min

    _format_xaxis(ax, show_xlabel, xlim_right=96)
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.12)
    _format_yaxis(ax, fmt_str=".3f")

    if show_ylabel:
        ax.set_ylabel(r"$\rho\, /\, \log_2 K$", fontsize=9, labelpad=6)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=2)


# ── Main ─────────────────────────────────────────────────────────────────

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

    fig = plt.figure(figsize=(14.5, 9.0))

    # Use nested gridspecs: outer splits TVSD | NSD with a gap,
    # inner grids handle the 2 columns within each dataset
    outer = gridspec.GridSpec(3, 2, figure=fig,
                              height_ratios=[0.40, 1, 1],
                              width_ratios=[1, 1],
                              hspace=0.38, wspace=0.18,
                              left=0.06, right=0.97, top=0.89, bottom=0.08)

    # Inner grids for each dataset pair (per-bit | raw)
    inner_tvsd_schem = outer[0, 0]
    inner_nsd_schem = outer[0, 1]
    inner_tvsd = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, 0],
                  wspace=0.35, width_ratios=[0.85, 1]) for row in (1, 2)]
    inner_nsd = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, 1],
                 wspace=0.35, width_ratios=[0.85, 1]) for row in (1, 2)]

    # ── Schematics (row 0) ──
    ax_tvsd_schem = fig.add_subplot(inner_tvsd_schem)
    draw_tvsd_schematic(ax_tvsd_schem)

    ax_nsd_schem = fig.add_subplot(inner_nsd_schem)
    draw_nsd_schematic(ax_nsd_schem)

    # ── Data panels (rows 1-2) ──
    # Map (row, col) to the correct inner gridspec cell
    panel_specs = [
        # (row, col, dataset, region, plot_fn, ylabel, xlabel, inner_grid, inner_col)
        (1, 0, "tvsd", "V1",                   plot_efficiency, True,  False, inner_tvsd[0], 0),
        (1, 1, "tvsd", "V1",                   plot_raw,        False, False, inner_tvsd[0], 1),
        (1, 2, "nsd",  "early visual stream",  plot_efficiency, False, False, inner_nsd[0],  0),
        (1, 3, "nsd",  "early visual stream",  plot_raw,        False, False, inner_nsd[0],  1),
        (2, 0, "tvsd", "IT",                   plot_efficiency, True,  True,  inner_tvsd[1], 0),
        (2, 1, "tvsd", "IT",                   plot_raw,        False, True,  inner_tvsd[1], 1),
        (2, 2, "nsd",  "ventral visual stream", plot_efficiency, False, True,  inner_nsd[1],  0),
        (2, 3, "nsd",  "ventral visual stream", plot_raw,        False, True,  inner_nsd[1],  1),
    ]

    axes = {}
    for row, col, ds, region, fn, ylabel, xlabel, inner, icol in panel_specs:
        ax = fig.add_subplot(inner[0, icol])
        fn(ax, ds, region, show_ylabel=ylabel, show_xlabel=xlabel)
        axes[(row, col)] = ax

    # ── Brain region insets on per-bit panels ──
    for row, col, brain_type, region_key in [
        (1, 0, "macaque", "V1"),    (2, 0, "macaque", "IT"),
        (1, 2, "human",   "early"), (2, 2, "human",   "ventral"),
    ]:
        add_brain_inset(axes[(row, col)], brain_type, region_key)

    # ── Column-pair headers (dataset name + stimulus type subtitle) ──
    for cols, schem_ax, title, subtitle in [
        ((0, 1), ax_tvsd_schem, "TVSD", "Object images"),
        ((2, 3), ax_nsd_schem,  "NSD",  "Natural scenes"),
    ]:
        left = axes[(1, cols[0])].get_position().x0
        right = axes[(1, cols[1])].get_position().x1
        x_center = (left + right) / 2
        y_top = schem_ax.get_position().y1
        fig.text(x_center, y_top + 0.035, title,
                 fontsize=14, fontweight="bold",
                 color="#1a1a1a", ha="center", va="bottom")
        fig.text(x_center, y_top + 0.015, subtitle,
                 fontsize=10, color="#777777", fontstyle="italic",
                 ha="center", va="bottom")

    # ── Row labels ──
    for row, label in [(1, "Early Visual\nCortex"), (2, "Higher Visual\nCortex")]:
        pos = axes[(row, 0)].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, label,
                 fontsize=10, fontweight="bold", color="#444444",
                 ha="center", va="center", rotation=90, linespacing=1.3)

    # ── Sub-column labels ──
    sub_labels = {
        (1, 0): "Alignment per bit",       (1, 1): "V1",
        (1, 2): "Alignment per bit",       (1, 3): "Early visual stream",
        # No headings for row 2 per-bit panels (e, g)
        (2, 1): "IT",
        (2, 3): "Ventral visual stream",
    }
    for key, label in sub_labels.items():
        pos = axes[key].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.018, label,
                 fontsize=9, color="#666666", ha="center", va="bottom")

    # ── Panel labels (a–h) ──
    for i, key in enumerate([(1, 0), (1, 1), (1, 2), (1, 3),
                             (2, 0), (2, 1), (2, 2), (2, 3)]):
        pos = axes[key].get_position()
        fig.text(pos.x0 - 0.015, pos.y1 + 0.028, chr(ord("a") + i),
                 fontsize=13, fontweight="bold", va="bottom", ha="left")

    # ── Vertical separator between TVSD and NSD ──
    # Use the gap between the two outer columns
    tvsd_right = axes[(1, 1)].get_position().x1
    nsd_left = axes[(1, 2)].get_position().x0
    sep_x = (tvsd_right + nsd_left) / 2
    bottom_y = axes[(2, 0)].get_position().y0 - 0.02
    top_y = ax_tvsd_schem.get_position().y1 + 0.01
    fig.add_artist(plt.Line2D(
        [sep_x, sep_x], [bottom_y, top_y],
        transform=fig.transFigure, color="#dddddd",
        linewidth=0.8, zorder=0))

    # ── Legend ──
    handles = [Line2D([], [], marker=ARCH_STYLE[k]["marker"], color="none",
                      markerfacecolor=ARCH_STYLE[k]["color"],
                      markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                      markersize=6, label=d)
               for k, _, d in ARCHITECTURES]
    axes[(1, 1)].legend(handles=handles, fontsize=7.5, frameon=True,
                        fancybox=False, framealpha=0.92, edgecolor="#dddddd",
                        borderpad=0.5, handletextpad=0.4, labelspacing=0.3,
                        title="Coarse label source", title_fontsize=7.5,
                        loc="lower right")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
