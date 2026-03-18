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

def plot_raw_coarseness(ax, dataset, region, show_ylabel=True, show_xlabel=True):
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
        # Light italic label above the line, left side, in front of everything
        y_offset = (y_max - y_min) * 0.03  # ~2mm visual offset above the line
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
        ax.set_xlabel("ImageNet training classes", fontsize=8, labelpad=8)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=8.5, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)

    _draw_bar_break(ax)

    # "(default)" subtitle below the 1000 tick — bottom row only
    if show_xlabel:
        ax.text(BAR_CENTER, -0.10, "(default)", fontsize=5.5,
                ha="center", va="top", color="#777777", fontstyle="italic",
                transform=ax.get_xaxis_transform())


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(11.5, 6.8))

    outer = gridspec.GridSpec(2, 3, figure=fig,
                              hspace=0.28, wspace=0.28,
                              height_ratios=[1, 1],
                              width_ratios=[0.7, 1, 1],
                              left=0.07, right=0.97, top=0.93, bottom=0.10)

    axes = {}

    # ── Row 0: TVSD ──
    ax_tvsd_schem = fig.add_subplot(outer[0, 0])
    draw_schematic_placeholder(ax_tvsd_schem,
                               "TVSD schematic\n(Macaque electrophysiology,\nV1 / V4 / IT)")
    axes[(0, 0)] = ax_tvsd_schem

    ax_v1 = fig.add_subplot(outer[0, 1])
    plot_raw_coarseness(ax_v1, "tvsd", "V1",
                        show_ylabel=True, show_xlabel=False)
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

    # ── Region titles ──
    for ax_key, title in [
        ((0, 1), "V1 (Early)"),
        ((0, 2), "IT (Late)"),
        ((1, 1), "Early Visual Stream"),
        ((1, 2), "Ventral Visual Stream"),
    ]:
        axes[ax_key].set_title(title, fontsize=9, fontweight="bold",
                                pad=6, color="#222222")

    # ── Row labels ──
    tvsd_mid_y = (axes[(0, 0)].get_position().y0 + axes[(0, 0)].get_position().y1) / 2
    nsd_mid_y = (axes[(1, 0)].get_position().y0 + axes[(1, 0)].get_position().y1) / 2
    fig.text(0.015, tvsd_mid_y, "TVSD  (Macaque)", fontsize=9.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)
    fig.text(0.015, nsd_mid_y, "NSD  (Human)", fontsize=9.5, fontweight="bold",
             ha="center", va="center", color="#333333", rotation=90)

    # ── Panel labels (A–F) ──
    label_order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, key in enumerate(label_order):
        label = chr(ord("A") + i)
        ax = axes[key]
        ax.text(-0.10, 1.10, label, transform=ax.transAxes,
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
        title="Latent repr. for\ncoarse labels",
        title_fontsize=7,
        loc="center left",
        bbox_to_anchor=(0.0, 0.45),
    )
    leg._legend_box.align = "left"

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
