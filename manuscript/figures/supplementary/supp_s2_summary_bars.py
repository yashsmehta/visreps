"""Supplementary Figure S2: Cross-Dataset Summary Bar Comparison.

Summary bar plots comparing pretrained models (ViT, CLIP) against
trained-from-scratch models (untrained, 1000-way, best coarse) for all
five dataset–region pairs.

Layout: 1×5 bar plots
  TVSD V1 | TVSD IT | NSD Early | NSD Ventral | THINGS

This was previously main Figure 5, now moved to supplementary.

Usage:
    python manuscript/figures/supplementary/supp_s2_summary_bars.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    setup_style, find_best_coarse_model, draw_no_data_bar,
    get_pretrained_summary,
)

OUTPUT = "manuscript/figures/supplementary/supp_s2_summary_bars.png"

# ── Color palette ──
BAR_COLORS = {
    "untrained": "#B0B0B0",
    "best_coarse": "#3274A1",
    "best_coarse_things": "#2CA02C",
    "1000way": "#E8963E",
    "vit": "#9467BD",
    "clip": "#17BECF",
}
BAR_HATCHES = {
    "vit": "", "clip": "",
    "untrained": "/", "1000way": "/", "best_coarse": "/",
}
HATCH_COLOR = "#777777"


def _draw_rounded_bar(ax, x, height, width, color, hatch="", zorder=3,
                      edgecolor="#555555", alpha=1.0):
    x0 = x - width / 2
    rect = mpatches.FancyBboxPatch(
        (x0, 0), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.012, rounding_size=0.05),
        facecolor=color, edgecolor=edgecolor, alpha=alpha,
        linewidth=0.5, hatch=hatch, mutation_aspect=0.04, zorder=zorder,
    )
    ax.add_patch(rect)


def _draw_bracket(ax, x_left, x_right, y_frac, label, color="#888888"):
    tick_h = 0.022
    for xp in [x_left, x_right]:
        ax.add_line(Line2D(
            [xp, xp], [y_frac, y_frac + tick_h],
            transform=ax.get_xaxis_transform(), color=color,
            linewidth=0.4, clip_on=False))
    ax.add_line(Line2D(
        [x_left, x_right], [y_frac, y_frac],
        transform=ax.get_xaxis_transform(), color=color,
        linewidth=0.4, clip_on=False, solid_capstyle="round"))
    ax.text((x_left + x_right) / 2, y_frac - 0.03, label,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=5, color="#555555")


def plot_summary_bars(ax, neural_dataset, region, title="", coarse_color=None,
                      show_ylabel=True):
    bar_width = 0.48
    gap = 0.42

    x_pre = np.array([0.0, 1.0])
    x_cnn = np.array([1.0 + gap + 1.0, 1.0 + gap + 2.0, 1.0 + gap + 3.0])
    x_positions = np.concatenate([x_pre, x_cnn])
    sep_x = (x_pre[-1] + x_cnn[0]) / 2

    conditions = ["ViT", "CLIP", "Untr.", "1K", "Coarse"]
    color_keys = ["vit", "clip", "untrained", "1000way", "best_coarse"]
    colors = [BAR_COLORS[k] for k in color_keys]
    if coarse_color is not None:
        colors[4] = coarse_color
    hatches = [BAR_HATCHES[k] for k in color_keys]

    vit_data = get_pretrained_summary(neural_dataset, region, "ViTBase")
    clip_data = get_pretrained_summary(neural_dataset, region, "CLIP_ViT_L14")
    un = get_condition_summary(neural_dataset, region, "imagenet1k", 1000,
                                "spearman", epoch=0, analysis="rsa")
    baseline = get_condition_summary(neural_dataset, region, "imagenet1k", 1000,
                                      "spearman", epoch=20, analysis="rsa")
    best = find_best_coarse_model(neural_dataset, region)
    values = [vit_data, clip_data, un, baseline, best]

    real_means = [v["mean"] for v in values if not np.isnan(v["mean"])]
    real_highs = [v.get("ci_high", v["mean"]) for v in values
                  if not np.isnan(v["mean"]) and not np.isnan(v.get("ci_high", v["mean"]))]
    if real_means:
        y_max = max(real_highs if real_highs else real_means) * 1.25
        ax.set_ylim(0, y_max)
    else:
        y_max = 0.1

    original_hatch_color = plt.rcParams.get("hatch.color")
    plt.rcParams["hatch.color"] = HATCH_COLOR

    pretrained_alpha = 0.42
    for i, (val, color, hatch) in enumerate(zip(values, colors, hatches)):
        mean = val["mean"]
        is_pretrained = i < 2
        alpha = pretrained_alpha if is_pretrained else 1.0
        if np.isnan(mean):
            placeholder_h = y_max * 0.25 if real_means else 0.05
            draw_no_data_bar(ax, x_positions[i], width=bar_width, color=color,
                             height=placeholder_h)
        else:
            edge = "#999999" if is_pretrained else "#555555"
            _draw_rounded_bar(ax, x_positions[i], mean, bar_width, color,
                              hatch=hatch, zorder=3, edgecolor=edge, alpha=alpha)
            ci_lo = val.get("ci_low", np.nan)
            ci_hi = val.get("ci_high", np.nan)
            if not np.isnan(ci_lo) and not np.isnan(ci_hi):
                err_lo = max(mean - ci_lo, 0)
                err_hi = max(ci_hi - mean, 0)
                err_color = "#999999" if is_pretrained else "#333333"
                if err_lo > 0 or err_hi > 0:
                    ax.errorbar(x_positions[i], mean, yerr=[[err_lo], [err_hi]],
                                fmt="none", ecolor=err_color, elinewidth=0.6,
                                capsize=2, capthick=0.6, zorder=5)

    if original_hatch_color is not None:
        plt.rcParams["hatch.color"] = original_hatch_color

    ax.axvline(sep_x, color="#C8C8C8", linestyle=(0, (2, 3)), linewidth=0.5,
               zorder=1, ymin=0.02, ymax=0.95)

    bracket_y = -0.16
    _draw_bracket(ax, x_pre[0], x_pre[1], bracket_y, "Pretrained")
    _draw_bracket(ax, x_cnn[0], x_cnn[2], bracket_y, "Trained from scratch")

    best_idx = 4
    if not np.isnan(best["mean"]):
        label_text = best.get("display_label", "")
        ci_hi = best.get("ci_high", best["mean"])
        y_annot = ci_hi if not np.isnan(ci_hi) else best["mean"]
        annot_color = "#1a6b30" if coarse_color else "#1a4a7a"
        ax.annotate(label_text, xy=(x_positions[best_idx], y_annot),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=4.8,
                    color=annot_color, fontstyle="italic", fontweight="medium")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, fontsize=5.5, ha="center")
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.tick_params(axis="x", direction="out", bottom=False, length=0, pad=2)
    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=7.5, labelpad=2)
    else:
        ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)

    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: "" if np.isclose(x, 0) else f"{x:.2f}"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="out", length=3,
                   width=0.5, labelsize=6)
    ax.tick_params(axis="y", which="minor", direction="out", length=1.8, width=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=3)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)


def main():
    setup_style()
    plt.rcParams.update({
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.size": 7,
    })

    fig = plt.figure(figsize=(14, 3.5))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.10,
                           top=0.82, bottom=0.12, left=0.05, right=0.98)
    axes_bars = [fig.add_subplot(gs[0, i]) for i in range(5)]

    THINGS_GREEN = BAR_COLORS["best_coarse_things"]

    panel_configs = [
        ("tvsd", "V1",                    "V1",              None),
        ("tvsd", "IT",                    "IT",              None),
        ("nsd",  "early visual stream",   "Early Visual",    None),
        ("nsd",  "ventral visual stream", "Ventral Visual",  None),
        ("things-behavior", "N/A",        "THINGS",          THINGS_GREEN),
    ]

    for idx, (ax, (nd, region, title, cc)) in enumerate(zip(axes_bars, panel_configs)):
        print(f"Drawing bars: {nd} / {region}")
        plot_summary_bars(ax, nd, region, title=title, coarse_color=cc,
                          show_ylabel=(idx == 0))
        if idx > 0:
            ax.set_yticklabels([])

    # Section headers & separators
    def _mid_x(ax):
        return ax.get_position().x0 + ax.get_position().width / 2

    def _gap_x(ax_left, ax_right):
        return (ax_left.get_position().x1 + ax_right.get_position().x0) / 2

    sep_tvsd_nsd = _gap_x(axes_bars[1], axes_bars[2])
    sep_nsd_things = _gap_x(axes_bars[3], axes_bars[4])

    tvsd_center = (_mid_x(axes_bars[0]) + _mid_x(axes_bars[1])) / 2
    nsd_center = (_mid_x(axes_bars[2]) + _mid_x(axes_bars[3])) / 2
    things_center = _mid_x(axes_bars[4])

    header_y = 0.945
    header_kw = dict(fontsize=9, fontweight="bold", ha="center", va="bottom",
                     color="#2c2c2c", fontfamily="sans-serif")
    fig.text(tvsd_center, header_y, "TVSD (Macaque)", **header_kw)
    fig.text(nsd_center, header_y, "NSD (Human fMRI)", **header_kw)
    fig.text(things_center, header_y, "THINGS (Behavioral)", **header_kw)

    for sx in [sep_tvsd_nsd, sep_nsd_things]:
        fig.add_artist(Line2D(
            [sx, sx], [0.08, 0.88],
            transform=fig.transFigure,
            color="#DCDCDC", linewidth=0.5, linestyle="-", zorder=0,
        ))

    for ax, label in zip(axes_bars, ["a", "b", "c", "d", "e"]):
        ax.text(-0.06, 1.12, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif")

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
