"""Extended CLIP coarseness curve for NSD neural alignment (2 → 1024).

Two-panel figure showing the full granularity sweep using CLIP-derived PCA
labels on the custom CNN (epoch 20) for early and ventral visual streams.
Uses a true log₂ x-axis (no axis break) since coarse labels extend to 1024.

Usage:
    python manuscript/figures/fig3/clip_extended_coarseness_nsd.py
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, AutoMinorLocator
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style

# ── Config ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig3"
COARSE_CFGS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
PCA_FOLDER = "pca_labels_clip"
REGIONS = [
    ("early visual stream", "Early visual stream"),
    ("ventral visual stream", "Ventral visual stream"),
]

# ── Colors (matching fig2/figure2.py) ────────────────────────────────────
CLIP_COLOR = "#08519c"
CLIP_FILL = "#4292c6"
BASELINE_1K_COLOR = "#e8963e"
UNTRAINED_COLOR = "#999999"


def fetch_region_data(region):
    """Fetch NSD RSA scores for all CLIP coarse granularities in a region."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("nsd", region, PCA_FOLDER, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def plot_panel(ax, region_key, region_label):
    """Plot one region's extended coarseness curve."""
    means, ci_lo, ci_hi = fetch_region_data(region_key)

    # Baselines
    bl = get_condition_summary("nsd", region_key, "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    un = get_condition_summary("nsd", region_key, "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")

    valid = ~np.isnan(means)
    x_valid = np.array(COARSE_CFGS)[valid]
    m_valid = means[valid]
    lo_valid = ci_lo[valid]
    hi_valid = ci_hi[valid]

    # Shaded CI band
    ax.fill_between(x_valid, lo_valid, hi_valid,
                    color=CLIP_FILL, alpha=0.18, zorder=2, edgecolor="none")

    # Connecting line
    ax.plot(x_valid, m_valid,
            color=CLIP_COLOR, linewidth=1.6, alpha=0.8, zorder=3)

    # Markers with error bars
    for i, cfg in enumerate(COARSE_CFGS):
        if np.isnan(means[i]):
            continue
        e_lo = max(means[i] - ci_lo[i], 0) if not np.isnan(ci_lo[i]) else 0
        e_hi = max(ci_hi[i] - means[i], 0) if not np.isnan(ci_hi[i]) else 0
        ax.errorbar(cfg, means[i], yerr=[[e_lo], [e_hi]],
                    fmt="o", color=CLIP_COLOR, markersize=5.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    capsize=2.5, capthick=0.6,
                    ecolor=CLIP_COLOR, elinewidth=0.7, zorder=5)

    # 1000-way baseline
    if not np.isnan(bl["mean"]):
        ax.axhline(bl["mean"], color=BASELINE_1K_COLOR, linewidth=1.0,
                   linestyle="--", alpha=0.85, zorder=1)
        ax.text(0.98, bl["mean"] + 0.003, "1000-way supervised",
                fontsize=7.5, fontstyle="italic", color=BASELINE_1K_COLOR,
                ha="right", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # Untrained baseline
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], color=UNTRAINED_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.7, zorder=1)
        ax.text(0.98, un["mean"] + 0.003, "Untrained",
                fontsize=7.5, fontstyle="italic", color=UNTRAINED_COLOR,
                ha="right", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # Axis formatting
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: str(int(v)) if v in COARSE_CFGS else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1.4, 1800)

    ax.set_xlabel("Number of training classes", fontsize=9.5, labelpad=5)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9.5, labelpad=4)
    ax.set_title(region_label, fontsize=10, fontweight="bold", pad=8)

    # Y-axis: tight range around data
    all_vals = list(m_valid) + list(lo_valid) + list(hi_valid)
    if not np.isnan(un["mean"]):
        all_vals.append(un["mean"])
    if not np.isnan(bl["mean"]):
        all_vals.append(bl["mean"])
    y_lo = np.nanmin(all_vals) - 0.02
    y_hi = np.nanmax(all_vals) + 0.02
    ax.set_ylim(y_lo, y_hi)

    ax.tick_params(axis="both", which="major", direction="out", labelsize=8.5)
    ax.tick_params(axis="x", which="major", rotation=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.yaxis.grid(True, which="major", color="#ECECEC", linewidth=0.4, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    sns.despine(ax=ax, right=True, top=True, offset=5)


def main():
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for ax, (region_key, region_label) in zip(axes, REGIONS):
        plot_panel(ax, region_key, region_label)

    fig.tight_layout(w_pad=3)
    out = f"{OUTPUT_DIR}/clip_extended_coarseness_nsd.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
