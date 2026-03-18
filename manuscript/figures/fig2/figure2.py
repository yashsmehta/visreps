"""Figure 2: Combined TVSD + NSD Neural Alignment.

2 rows × 3 columns:
  Column 0: dataset schematics (placeholders)
  Columns 1-2: data panels
  Row 0 (TVSD):  Schematic | V1 coarseness | IT coarseness
  Row 1 (NSD):   Schematic | Early coarseness | Ventral coarseness

Coarseness panels show raw Spearman ρ values.
AlexNet/CLIP as blue shades, Pixels as brown.
Untrained + 1000-way shown as grouped bar pair after axis break.

Usage:
    python manuscript/figures/fig2/figure2.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH,
    setup_style, compute_jitter, draw_schematic_placeholder,
)

OUTPUT_DIR = "manuscript/figures/fig2"

# ── Color scheme ─────────────────────────────────────────────────────────
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
UNTRAINED_BAR_COLOR = "#999999"  # medium gray

# ── Grouped bar positions (log₂ axis) ────────────────────────────────────
BAR_CENTER = 250
BAR_LEFT = BAR_CENTER / 1.16    # untrained  (~216)
BAR_RIGHT = BAR_CENTER * 1.16   # trained    (~290)
BAR_WIDTH_FRAC = 0.15


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
    return s["mean"], s["ci_low"], s["ci_high"]


def _sem_summary(df):
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
    return mean, sem


def fetch_tvsd_arch_data(folder, region):
    means, sems = [], []
    for cfg in COARSE_CFGS:
        df = query_best_scores("tvsd", region, folder, cfg,
                               "spearman", epoch=20, analysis="rsa")
        if df.empty:
            means.append(np.nan)
            sems.append(0)
            continue
        m, s = _sem_summary(df)
        means.append(m)
        sems.append(s)
    return np.array(means), np.array(sems)


def fetch_tvsd_baseline(region, epoch=20):
    df = query_best_scores("tvsd", region, "imagenet1k", 1000,
                           "spearman", epoch=epoch, analysis="rsa")
    if df.empty:
        return np.nan, np.nan, np.nan
    m, s = _sem_summary(df)
    return m, m - 1.96 * s, m + 1.96 * s


# ── Axis break ───────────────────────────────────────────────────────────

def _draw_bar_break(ax):
    """Draw // break marks between the coarse scatter region and the bars."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mid = np.exp((np.log(64) + np.log(BAR_LEFT)) / 2)
    rect_hw = mid * 0.16
    rect = mpatches.FancyBboxPatch(
        (mid / 1.16, -0.05), width=rect_hw * 1.5, height=0.10,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none",
        transform=trans, clip_on=False, zorder=9)
    ax.add_patch(rect)
    for x_shift in [0.93, 1.07]:
        x_c = mid * x_shift
        ax.plot([x_c / 1.04, x_c * 1.04], [-0.028, 0.028],
                transform=trans, color="#555555", linewidth=0.7,
                clip_on=False, zorder=11)


# ── Tick label formatter ─────────────────────────────────────────────────

def _make_tick_formatter(label_map):
    """Tolerance-based tick formatter for log-axis tick matching."""
    def _fmt(val, pos):
        for k, lbl in label_map.items():
            if abs(val - k) < k * 0.05:
                return lbl
        return ""
    return _fmt


# ── Coarseness panel ─────────────────────────────────────────────────────

def plot_raw_coarseness(ax, dataset, region, show_ylabel=True, show_xlabel=True,
                        show_untrained_label=True):
    """Plot coarseness panel with raw Spearman ρ values."""

    if dataset == "nsd":
        bl_mean, bl_ci_lo, bl_ci_hi = fetch_nsd_baseline(region, epoch=20)
        un_mean, un_ci_lo, un_ci_hi = fetch_nsd_baseline(region, epoch=0)
    else:
        bl_mean, bl_ci_lo, bl_ci_hi = fetch_tvsd_baseline(region, epoch=20)
        un_mean, un_ci_lo, un_ci_hi = fetch_tvsd_baseline(region, epoch=0)

    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    # ── 1) Architecture scatter points ──
    all_y_vals = [bl_mean]
    if not np.isnan(un_mean):
        all_y_vals.append(un_mean)

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

    # ── 2) Y-axis range (no forced zero) ──
    y_min = min(all_y_vals)
    y_max = max(all_y_vals)
    y_range = y_max - y_min
    y_bottom = y_min - y_range * 0.12

    # ── 3) Untrained dashed line (mean only, behind everything) + label ──
    if not np.isnan(un_mean):
        ax.axhline(un_mean, color="#AAAAAA", linestyle="--",
                    linewidth=0.9, alpha=0.7, zorder=1)
        if show_untrained_label:
            y_offset = (y_max - y_min) * 0.03
            ax.text(0.02, un_mean + y_offset, " Untrained",
                    fontsize=6, fontstyle="italic", color="#AAAAAA",
                    ha="left", va="bottom",
                    transform=blended_transform_factory(ax.transAxes, ax.transData),
                    zorder=10)

    # ── 4) Single 1000-way trained bar ──
    bl_err_lo = max(bl_mean - bl_ci_lo, 0) if not np.isnan(bl_ci_lo) else 0
    bl_err_hi = max(bl_ci_hi - bl_mean, 0) if not np.isnan(bl_ci_hi) else 0
    ax.bar(BAR_CENTER, bl_mean - y_bottom, bottom=y_bottom,
           width=BAR_CENTER * BAR_WIDTH_FRAC,
           color=BASELINE_1K_COLOR, edgecolor="#c07830",
           linewidth=0.4, zorder=3)
    ax.errorbar(BAR_CENTER, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="none", ecolor="#555555", elinewidth=0.7,
                capsize=2.2, capthick=0.6, zorder=5)

    # ── 4) Axis formatting ──
    ax.set_xscale("log", base=2)

    # Tick labels: coarse ticks + "1000" at BAR_CENTER
    all_ticks = COARSE_CFGS + [BAR_CENTER]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BAR_CENTER] = "1000"
    ax.xaxis.set_major_locator(FixedLocator(all_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_make_tick_formatter(label_map)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    # Hide the tick *mark* at BAR_CENTER (label only, no spine notch)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6)
    if not show_xlabel:
        ax.set_xticklabels([""] * len(all_ticks))
    ax.set_xlim(1.5, BAR_CENTER * 1.35)

    # Y-axis
    ax.tick_params(axis="y", which="major", direction="out", length=3.5, width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    # Small top margin
    cur_ylim = ax.get_ylim()
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + y_range * 0.03)

    if show_xlabel:
        ax.set_xlabel("ImageNet training classes", fontsize=9, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)

    _draw_bar_break(ax)

    # "(default)" subtitle below the 1000 tick — removed for cleanliness


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

    fig = plt.figure(figsize=(11.5, 7.2))

    outer = gridspec.GridSpec(2, 3, figure=fig,
                              hspace=0.42, wspace=0.28,
                              height_ratios=[1, 1],
                              width_ratios=[0.7, 1, 1],
                              left=0.08, right=0.97, top=0.88, bottom=0.10)

    axes = {}

    # ── Row 0: TVSD ──
    ax_tvsd_schem = fig.add_subplot(outer[0, 0])
    draw_schematic_placeholder(ax_tvsd_schem,
                               "TVSD schematic\n(Macaque electrophysiology,\nV1 / V4 / IT)")
    axes[(0, 0)] = ax_tvsd_schem

    ax_v1 = fig.add_subplot(outer[0, 1])
    plot_raw_coarseness(ax_v1, "tvsd", "V1",
                        show_ylabel=True, show_xlabel=False,
                        show_untrained_label=True)
    axes[(0, 1)] = ax_v1

    ax_it = fig.add_subplot(outer[0, 2])
    plot_raw_coarseness(ax_it, "tvsd", "IT",
                        show_ylabel=False, show_xlabel=False)
    axes[(0, 2)] = ax_it

    # ── Row 1: NSD ──
    ax_nsd_schem = fig.add_subplot(outer[1, 0])
    draw_schematic_placeholder(ax_nsd_schem,
                               "NSD schematic\n(Human fMRI,\n8 subjects, early/ventral\nvisual stream)")
    axes[(1, 0)] = ax_nsd_schem

    ax_early = fig.add_subplot(outer[1, 1])
    plot_raw_coarseness(ax_early, "nsd", "early visual stream",
                        show_ylabel=True, show_xlabel=True)
    axes[(1, 1)] = ax_early

    ax_ventral = fig.add_subplot(outer[1, 2])
    plot_raw_coarseness(ax_ventral, "nsd", "ventral visual stream",
                        show_ylabel=False, show_xlabel=True)
    axes[(1, 2)] = ax_ventral

    # ── Column headers (shared across both rows) ──
    # One bold title per column, positioned above the TVSD (top) row.
    # Per-row gray subtitles list the specific brain regions.
    col_headers = {
        1: "Early Visual Cortex",
        2: "Higher Visual Cortex",
    }
    for col, header in col_headers.items():
        pos = axes[(0, col)].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        fig.text(x_center, y_top + 0.058, header,
                 fontsize=11.5, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom", family="sans-serif")

    # Per-row subtitles: stream/region name + specific ROIs in brackets
    # TVSD: single region names (V1, IT)
    # NSD: stream name + constituent ROIs
    row_subtitles = {
        # (ax_key): (line1, line2_or_None)
        (0, 1): ("V1", None),
        (0, 2): ("IT", None),
        (1, 1): ("Early visual stream", "(V1, V2, V3)"),
        (1, 2): ("Ventral visual stream", "(VO, PHC, and higher areas)"),
    }
    for ax_key, (line1, line2) in row_subtitles.items():
        ax = axes[ax_key]
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_top = pos.y1
        if line2:
            # Two-line subtitle: stream name + ROIs in brackets
            fig.text(x_center, y_top + 0.025, line1,
                     fontsize=8, color="#888888",
                     ha="center", va="bottom", family="sans-serif")
            fig.text(x_center, y_top + 0.005, line2,
                     fontsize=6.5, color="#aaaaaa",
                     ha="center", va="bottom", family="sans-serif")
        else:
            # Single-line subtitle (TVSD)
            fig.text(x_center, y_top + 0.012, line1,
                     fontsize=8, color="#888888",
                     ha="center", va="bottom", family="sans-serif")

    # ── Row labels (two-level: bold dataset + lighter species) ──
    # Use two separate x positions to stack them side-by-side when rotated 90°
    tvsd_mid_y = (axes[(0, 0)].get_position().y0 + axes[(0, 0)].get_position().y1) / 2
    nsd_mid_y = (axes[(1, 0)].get_position().y0 + axes[(1, 0)].get_position().y1) / 2
    # TVSD — bold name at x=0.010, species at x=0.028 (stacks left-to-right)
    fig.text(0.008, tvsd_mid_y, "TVSD", fontsize=10, fontweight="bold",
             ha="center", va="center", color="#2a2a2a", rotation=90)
    fig.text(0.028, tvsd_mid_y, "Macaque", fontsize=7.5,
             ha="center", va="center", color="#888888", rotation=90)
    # NSD
    fig.text(0.008, nsd_mid_y, "NSD", fontsize=10, fontweight="bold",
             ha="center", va="center", color="#2a2a2a", rotation=90)
    fig.text(0.028, nsd_mid_y, "Human", fontsize=7.5,
             ha="center", va="center", color="#888888", rotation=90)

    # ── Panel labels (A–F) ──
    label_order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, key in enumerate(label_order):
        label = chr(ord("A") + i)
        ax = axes[key]
        # Top-row data panels: extra offset for column header + subtitle
        # Bottom-row data panels: offset for subtitle only
        # Schematic panels (col 0): standard offset
        if key[1] == 0:
            y_offset = 1.10
        elif key[0] == 0:
            y_offset = 1.30  # top row with column header above
        else:
            y_offset = 1.16  # bottom row with subtitle only
        ax.text(-0.10, y_offset, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # ── Remove per-axes legends (except panel B) ──
    for key, ax in axes.items():
        if key == (0, 1):
            continue
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # ── In-plot legend in TVSD V1 (panel B) ──
    arch_handles = []
    for arch_key, _, display in ARCHITECTURES:
        style = ARCH_STYLE[arch_key]
        h = Line2D([], [], marker=style["marker"], color="none",
                   markerfacecolor=style["color"],
                   markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                   markersize=5.5, label=display)
        arch_handles.append(h)
    leg = axes[(0, 1)].legend(
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

    # ── Subtle row separator ──
    tvsd_bottom = axes[(0, 0)].get_position().y0
    nsd_top_title = axes[(1, 1)].get_position().y1 + 0.060
    sep_y = (tvsd_bottom + nsd_top_title) / 2
    fig.add_artist(plt.Line2D(
        [0.04, 0.98], [sep_y, sep_y],
        transform=fig.transFigure, color="#cccccc",
        linewidth=0.8, zorder=0))

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
