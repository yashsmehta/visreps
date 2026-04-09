"""Panel renderer: raw Spearman rho scatter with broken x-axis for all label sources."""

import sys

import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, "manuscript/figures/fig3")
from shared import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, compute_jitter,
    ARCHITECTURES, ARCH_STYLE, BASELINE_1K_COLOR,
    fetch_arch_data, fetch_baseline, fetch_baseline_ci, format_yaxis,
)

sys.path.insert(0, "manuscript/figures")
from fig_utils import BREAK_1K_POS, draw_xaxis_break


def _format_broken_xaxis(ax, show_xlabel):
    """Log-2 x-axis with broken gap before 1000-way position."""
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


def plot_raw(ax, dataset, region, show_ylabel=True, show_xlabel=True,
             show_untrained_label=False, tick_interval=None, lollipop_ax=None):
    """Raw Spearman rho scatter (all architectures) + 1000-way marker + broken axis."""
    bl_mean, bl_ci_low, bl_ci_high = fetch_baseline_ci(dataset, region)
    if np.isnan(bl_mean) or bl_mean == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#888")
        return

    untrained_mean = fetch_baseline(dataset, region, epoch=0)

    all_y = [bl_mean]
    if not np.isnan(bl_ci_low):
        all_y.append(bl_ci_low)
    if not np.isnan(bl_ci_high):
        all_y.append(bl_ci_high)
    if not np.isnan(untrained_mean):
        all_y.append(untrained_mean)

    # Coarse conditions (2–64)
    for arch_idx, (arch_key, folder, _) in enumerate(ARCHITECTURES):
        style = ARCH_STYLE[arch_key]
        means, errs_lo, errs_hi = fetch_arch_data(dataset, folder, region)
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

    # 1000-way CI band: light orange horizontal span (full extent; torn later)
    if not np.isnan(bl_ci_low) and not np.isnan(bl_ci_high):
        ax.fill_between([1.5, BREAK_1K_POS], bl_ci_low, bl_ci_high,
                        facecolor=BASELINE_1K_COLOR, alpha=0.12,
                        edgecolor="none", zorder=1)

    # Dashed reference line at 1000-way level (full extent; torn later)
    ax.plot([1.5, BREAK_1K_POS], [bl_mean, bl_mean],
            color=BASELINE_1K_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.6, zorder=2, clip_on=False)

    # 1000-way baseline: orange diamond at broken-axis position
    bl_err_lo = max(bl_mean - bl_ci_low, 0) if not np.isnan(bl_ci_low) else 0
    bl_err_hi = max(bl_ci_high - bl_mean, 0) if not np.isnan(bl_ci_high) else 0
    ax.errorbar(BREAK_1K_POS, bl_mean,
                yerr=[[bl_err_lo], [bl_err_hi]],
                fmt="D", color=BASELINE_1K_COLOR, markersize=MARKER_SIZE,
                markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                capsize=1.5, capthick=0.5,
                ecolor=BASELINE_1K_COLOR, elinewidth=0.7, zorder=5)

    # Untrained baseline (zorder=3 so it paints over the orange tear mask below)
    if not np.isnan(untrained_mean):
        ax.axhline(untrained_mean, color="#AAAAAA", linestyle="--",
                    linewidth=1.25, alpha=0.6, zorder=3)
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

    # ── Jagged tear through the orange band + dashed line at the break ──
    # Reuses the exact tooth dimensions from _draw_lollipop_break in panel_bits:
    # - x protrusion: 0.010 axes-fraction x (scatter shares x plot-area with
    #   lollipop, so the tooth has the same pixel width as the gray break).
    # - y period:  0.15 axes-fraction y of the lollipop panel, scaled by
    #   (lollipop_height / scatter_height) so each tooth has the same pixel
    #   height as the gray break.
    # Number of teeth adapts to the orange band's height so each tooth stays
    # a fixed visual size regardless of how wide the CI is.
    from matplotlib.patches import Polygon
    from matplotlib.transforms import blended_transform_factory
    import math

    GRAY_TOOTH_DX = 0.010          # axes-fraction x (reused as-is)
    GRAY_SEGMENT_Y_PERIOD = 0.15   # axes-fraction y of the lollipop panel

    # Center of the break, in scatter axes fraction, then shifted left to
    # coincide with the LEFT jagged edge of the gray lollipop break
    # (gap=0.045 in panel_bits._draw_lollipop_break, so left edge = center - gap/2).
    LOLLIPOP_BREAK_GAP = 0.045  # must match panel_bits._draw_lollipop_break
    x_lo, x_hi = 1.5, BREAK_1K_POS * 1.5
    mid_data_center = math.exp((math.log(64) + math.log(BREAK_1K_POS)) / 2)
    center_frac = (math.log2(mid_data_center) - math.log2(x_lo)) / (math.log2(x_hi) - math.log2(x_lo))
    mid_frac = center_frac - LOLLIPOP_BREAK_GAP / 2

    # Scale tooth y-period from lollipop axes-fraction to scatter data units
    if lollipop_ax is not None:
        height_ratio = lollipop_ax.get_position().height / ax.get_position().height
    else:
        height_ratio = 0.14 / 0.86   # fallback matching figure3.py height_ratios
    y_lo_lim, y_hi_lim = ax.get_ylim()
    tooth_y_period = GRAY_SEGMENT_Y_PERIOD * height_ratio * (y_hi_lim - y_lo_lim)

    # ── Fit an integer number of segments exactly into the band ──
    # Rounding band_height to an integer multiple of tooth_y_period means
    # the visible zigzag terminates at clean tooth tips on both band edges.
    # Minimum 4 segments (2 full teeth) so even tight bands show a jag.
    band_height = bl_ci_high - bl_ci_low
    n_seg_band = max(4, int(round(band_height / tooth_y_period)))
    seg_period = band_height / n_seg_band  # adjusted so it divides evenly

    # Preserve tooth aspect ratio: if band is too narrow for the standard
    # tooth_y_period, scale x protrusion down by the same factor so each
    # tooth is a miniaturised (similar) version of the reference tooth
    # rather than a squished one. Never scale up (cap ratio at 1).
    aspect_scale = min(1.0, seg_period / tooth_y_period)
    vis_tooth_dx = GRAY_TOOTH_DX * aspect_scale

    # Blended transform: x in axes fraction, y in data coordinates
    trans_bt = blended_transform_factory(ax.transAxes, ax.transData)

    # ── Visible orange zigzag: exactly spans [bl_ci_low, bl_ci_high] ──
    y_vis = bl_ci_low + np.arange(n_seg_band + 1) * seg_period
    zigzag_vis = np.array([(-1) ** k for k in range(len(y_vis))])
    x_vis = mid_frac + vis_tooth_dx * zigzag_vis

    # ── White mask: extend the same zigzag past plot edges (same phase) ──
    # Keep the mask tearing the dashed line everywhere outside the band.
    n_ext = max(2, int(math.ceil((y_hi_lim - y_lo_lim) / seg_period)))
    # Segments above the band: alternate sign continuing from the top tip
    y_above = bl_ci_high + np.arange(1, n_ext + 1) * seg_period
    zig_above = np.array([(-1) ** (n_seg_band + k) for k in range(1, n_ext + 1)])
    x_above = mid_frac + vis_tooth_dx * zig_above
    # Segments below the band: alternate sign continuing from the bottom tip
    y_below = bl_ci_low - np.arange(1, n_ext + 1) * seg_period
    zig_below = np.array([(-1) ** (-k) for k in range(1, n_ext + 1)])
    x_below = mid_frac + vis_tooth_dx * zig_below

    y_mask_pts = np.concatenate([y_below[::-1], y_vis, y_above])
    x_mask_pts = np.concatenate([x_below[::-1], x_vis, x_above])

    right_far = 1.15
    verts = (list(zip(x_mask_pts, y_mask_pts))
             + [(right_far, y_mask_pts[-1]), (right_far, y_mask_pts[0])])
    ax.add_patch(Polygon(verts, facecolor="white", edgecolor="none",
                         transform=trans_bt, clip_on=True, zorder=2.6))

    # Orange outline: only the in-band portion, clean tooth-tip endpoints
    ax.plot(x_vis, y_vis, transform=trans_bt,
            color=BASELINE_1K_COLOR, linewidth=0.6, alpha=0.75,
            solid_joinstyle="miter", solid_capstyle="butt",
            clip_on=False, zorder=2.7)

    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)
