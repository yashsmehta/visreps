"""Extended CLIP coarseness curve for THINGS behavioral alignment (2 → 1024).

Standalone figure showing the full granularity sweep using CLIP-derived PCA
labels on the custom CNN (epoch 20). Unlike Panel B of Figure 3, this uses a
true log₂ x-axis (no axis break) since coarse labels extend to 1024.

Usage:
    python manuscript/figures/fig3/clip_extended_coarseness.py
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

# ── Colors ───────────────────────────────────────────────────────────────
CLIP_COLOR = "#08519c"
CLIP_FILL = "#4292c6"
BASELINE_1K_COLOR = "#d4822e"
UNTRAINED_COLOR = "#999999"


def fetch_clip_data():
    """Fetch THINGS RSA scores for all CLIP coarse granularities."""
    means, ci_lo, ci_hi = [], [], []
    for cfg in COARSE_CFGS:
        s = get_condition_summary("things-behavior", "N/A", PCA_FOLDER, cfg,
                                  "spearman", epoch=20, analysis="rsa")
        means.append(s["mean"])
        ci_lo.append(s["ci_low"])
        ci_hi.append(s["ci_high"])
    return np.array(means), np.array(ci_lo), np.array(ci_hi)


def main():
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # ── Fetch data ────────────────────────────────────────────────────────
    means, ci_lo, ci_hi = fetch_clip_data()

    # 1000-way baseline
    bl = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=20, analysis="rsa")
    # Untrained baseline
    un = get_condition_summary("things-behavior", "N/A", "imagenet1k", 1000,
                               "spearman", epoch=0, analysis="rsa")

    valid = ~np.isnan(means)
    x_valid = np.array(COARSE_CFGS)[valid]
    m_valid = means[valid]
    lo_valid = ci_lo[valid]
    hi_valid = ci_hi[valid]

    # ── Shaded CI band ────────────────────────────────────────────────────
    ax.fill_between(x_valid, lo_valid, hi_valid,
                    color=CLIP_FILL, alpha=0.18, zorder=2,
                    edgecolor="none")

    # ── Connecting line ───────────────────────────────────────────────────
    ax.plot(x_valid, m_valid,
            color=CLIP_COLOR, linewidth=1.6, alpha=0.8, zorder=3)

    # ── Markers with error bars ───────────────────────────────────────────
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

    # ── 1000-way baseline ─────────────────────────────────────────────────
    if not np.isnan(bl["mean"]):
        ax.axhline(bl["mean"], color=BASELINE_1K_COLOR, linewidth=1.0,
                   linestyle="--", alpha=0.85, zorder=1)
        ax.text(0.98, bl["mean"] + 0.005, "1000-way supervised",
                fontsize=7.5, fontstyle="italic", color=BASELINE_1K_COLOR,
                ha="right", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # ── Untrained baseline ────────────────────────────────────────────────
    if not np.isnan(un["mean"]):
        ax.axhline(un["mean"], color=UNTRAINED_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.7, zorder=1)
        ax.text(0.98, un["mean"] + 0.005, "Untrained",
                fontsize=7.5, fontstyle="italic", color=UNTRAINED_COLOR,
                ha="right", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(COARSE_CFGS))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: str(int(v)) if v in COARSE_CFGS else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1.4, 1800)

    ax.set_xlabel("Number of training classes", fontsize=9.5, labelpad=5)
    ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9.5, labelpad=4)

    # Y-axis: tight range around data with breathing room
    y_pad_lo = 0.04
    y_pad_hi = 0.03
    y_lo = un["mean"] - y_pad_lo if not np.isnan(un["mean"]) else 0.15
    y_hi = np.nanmax(hi_valid) + y_pad_hi
    ax.set_ylim(y_lo, y_hi)

    ax.tick_params(axis="both", which="major", direction="out", labelsize=8.5)
    ax.tick_params(axis="x", which="major", rotation=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out")
    ax.yaxis.grid(True, which="major", color="#ECECEC", linewidth=0.4, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    sns.despine(ax=ax, right=True, top=True, offset=5)

    fig.tight_layout()
    out = f"{OUTPUT_DIR}/clip_extended_coarseness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
