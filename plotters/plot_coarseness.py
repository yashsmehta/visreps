"""Unified coarseness progression plotter.

Generates two figures per (dataset, architecture) combination:
  1. Coarseness bars — grand mean across seeds+subjects, bootstrap 95% CIs
     (SEM fallback when bootstrap unavailable). Includes untrained reference
     and axis break between 64 and 1000.
  2. Per-subject boxes — individual subjects (averaged across seeds) connected
     by lines. Skipped for THINGS (no subjects).

Usage:
    python plotters/plot_coarseness.py --dataset nsd   --folder pca_labels_alexnet
    python plotters/plot_coarseness.py --dataset tvsd  --folder pca_labels_alexnet
    python plotters/plot_coarseness.py --dataset things --folder pca_labels_alexnet
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, get_subject_scores

# ── Dataset configuration ─────────────────────────────────────────────────
DATASET_CONFIG = {
    "nsd": {
        "neural_dataset": "nsd",
        "regions": ["early visual stream", "ventral visual stream"],
        "region_labels": {
            "early visual stream": "Early Visual Stream",
            "ventral visual stream": "Ventral Visual Stream",
        },
        "has_subjects": True,
    },
    "tvsd": {
        "neural_dataset": "tvsd",
        "regions": ["V1", "V4", "IT"],
        "region_labels": {"V1": "V1", "V4": "V4", "IT": "IT"},
        "has_subjects": True,
    },
    "things": {
        "neural_dataset": "things-behavior",
        "regions": ["N/A"],
        "region_labels": {"N/A": "THINGS Behavior"},
        "has_subjects": False,
    },
}

FOLDER_DISPLAY = {
    "pca_labels_alexnet": "AlexNet",
    "pca_labels_vit": "ViT",
    "pca_labels_clip": "CLIP",
    "pca_labels_dino": "DINO",
}

COARSE_CFGS = [2, 4, 8, 16, 32, 64]
N_COARSE = len(COARSE_CFGS)

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams["hatch.color"] = "grey"

blues = sns.color_palette("Blues", n_colors=N_COARSE + 1)[1:]
UNTRAINED_COLOR = "#AAAAAA"
BASELINE_COLOR = "#FFA500"
BAR_WIDTH = 0.72


# ── Fancy bar helpers ─────────────────────────────────────────────────────
def draw_fancy_bar(ax, x, height, color, hatch="", width=BAR_WIDTH):
    """Draw a bar using FancyBboxPatch with subtle top rounding."""
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
        facecolor=color, edgecolor="black", linewidth=0.8,
        hatch=hatch, mutation_aspect=0.05, zorder=3,
    )
    ax.add_patch(rect)


def draw_break_marks(ax, x):
    """Draw diagonal slashes at the bottom spine to indicate axis break."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    spine_y, dy, dx, gap = -0.022, 0.028, 0.20, 0.13
    ax.plot([x - gap - dx - 0.1, x + gap + dx + 0.1], [spine_y, spine_y],
            color="white", linewidth=5, transform=trans, clip_on=False, zorder=9)
    for offset in [-gap, gap]:
        ax.plot([x + offset - dx, x + offset + dx], [spine_y - dy, spine_y + dy],
                color="black", linewidth=1.8, clip_on=False, zorder=10,
                transform=trans)


# ── Figure 1: Coarseness bars ────────────────────────────────────────────
def plot_coarseness_bars(args, dcfg):
    """Fancy bar plot: untrained | coarse (2-64) | break | 1000."""
    nd = dcfg["neural_dataset"]
    regions = dcfg["regions"]
    n_regions = len(regions)
    display_name = FOLDER_DISPLAY.get(args.folder, args.folder)

    fig, axes = plt.subplots(1, n_regions,
                             figsize=(5 * n_regions, 4), sharey=False,
                             squeeze=False)

    for idx, region in enumerate(regions):
        ax = axes[0, idx]

        # Check for untrained data (epoch=0)
        un = get_condition_summary(nd, region, "imagenet1k", 1000,
                                   "spearman", epoch=0)
        has_untrained = not np.isnan(un["mean"])

        # Build X positions
        if has_untrained:
            X_COARSE = np.arange(1.5, 1.5 + N_COARSE)
            X_UNTRAINED = 0.0
        else:
            X_COARSE = np.arange(N_COARSE, dtype=float)
            X_UNTRAINED = None
        X_BASELINE = X_COARSE[-1] + 2
        BREAK_X = (X_COARSE[-1] + X_BASELINE) / 2

        # Collect means + CIs
        all_means, all_ci_lo, all_ci_hi = [], [], []
        all_x, all_colors, all_hatches, all_labels = [], [], [], []

        if has_untrained:
            all_means.append(un["mean"])
            all_ci_lo.append(un["ci_low"])
            all_ci_hi.append(un["ci_high"])
            all_x.append(X_UNTRAINED)
            all_colors.append(UNTRAINED_COLOR)
            all_hatches.append("")
            all_labels.append("Untrained")

        for i, cfg in enumerate(COARSE_CFGS):
            s = get_condition_summary(nd, region, args.folder, cfg,
                                      "spearman", epoch=20)
            all_means.append(s["mean"])
            all_ci_lo.append(s["ci_low"])
            all_ci_hi.append(s["ci_high"])
            all_x.append(X_COARSE[i])
            all_colors.append(blues[i])
            all_hatches.append("/")
            all_labels.append(str(cfg))

        bl = get_condition_summary(nd, region, "imagenet1k", 1000,
                                   "spearman", epoch=20)
        all_means.append(bl["mean"])
        all_ci_lo.append(bl["ci_low"])
        all_ci_hi.append(bl["ci_high"])
        all_x.append(X_BASELINE)
        all_colors.append(BASELINE_COLOR)
        all_hatches.append("")
        all_labels.append("1000")

        all_means = np.array(all_means)
        all_ci_lo = np.array(all_ci_lo)
        all_ci_hi = np.array(all_ci_hi)
        all_x = np.array(all_x)
        err_lo = all_means - all_ci_lo
        err_hi = all_ci_hi - all_means

        # Y-axis range
        valid = ~np.isnan(all_means)
        if valid.any():
            vlo = np.nanmin(all_ci_lo[~np.isnan(all_ci_lo)]) if (~np.isnan(all_ci_lo)).any() else np.nanmin(all_means)
            vhi = np.nanmax(all_ci_hi[~np.isnan(all_ci_hi)]) if (~np.isnan(all_ci_hi)).any() else np.nanmax(all_means)
        else:
            vlo, vhi = 0, 0.1
        dr = max(vhi - vlo, 0.01)
        y_bottom = max(0, vlo - 0.20 * dr)
        y_top = vhi + 0.20 * dr

        # Draw fancy bars
        for k in range(len(all_x)):
            if not np.isnan(all_means[k]):
                draw_fancy_bar(ax, all_x[k], all_means[k],
                               all_colors[k], all_hatches[k])

        # Error bars (both must be non-negative for matplotlib)
        for k in range(len(all_x)):
            if (not np.isnan(err_lo[k]) and not np.isnan(err_hi[k])
                    and err_lo[k] >= 0 and err_hi[k] >= 0
                    and (err_lo[k] > 0 or err_hi[k] > 0)):
                ax.errorbar(all_x[k], all_means[k],
                            yerr=[[err_lo[k]], [err_hi[k]]],
                            fmt="none", ecolor="black", elinewidth=1.0,
                            capsize=4, capthick=1.0, zorder=5)

        draw_break_marks(ax, BREAK_X)

        # Axis formatting
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels, fontsize=10, ha="center")
        ax.tick_params(axis="x", direction="out", bottom=False, length=4, width=1.5)
        ax.tick_params(axis="y", which="major", direction="out", labelsize=12,
                       length=5, width=1.5)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", direction="out", length=3, width=1.0)
        first_x = X_UNTRAINED if has_untrained else all_x[0]
        ax.set_xlim(first_x - 0.6, X_BASELINE + 0.7)
        ax.set_ylim(y_bottom, y_top)

        ax.set_xlabel("Number of Classes", fontsize=13)
        ax.set_ylabel("Spearman \u03c1", fontsize=13)
        region_label = dcfg["region_labels"].get(region, region)
        ax.set_title(region_label, fontsize=15, fontweight="bold", pad=10)

        sns.despine(ax=ax, right=True, top=True, offset=5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

    fig.suptitle(
        f"Brain Alignment Across Label Granularity\n"
        f"({display_name}-PCA Labels, {args.dataset.upper()} RSA)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=1.0)
    out = f"plotters/figures/{args.dataset}/coarseness_bars_{display_name.lower()}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


# ── Figure 2: Per-subject boxes ──────────────────────────────────────────
def plot_per_subject(args, dcfg):
    """Box plots with per-subject dots connected across class counts."""
    if not dcfg["has_subjects"]:
        print(f"Skipping per-subject plot ({args.dataset} has no subjects)")
        return

    nd = dcfg["neural_dataset"]
    regions = dcfg["regions"]
    n_regions = len(regions)
    display_name = FOLDER_DISPLAY.get(args.folder, args.folder)

    fig, axes = plt.subplots(1, n_regions,
                             figsize=(5 * n_regions, 4), sharey=False,
                             squeeze=False)

    blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]

    for idx, region in enumerate(regions):
        ax = axes[0, idx]
        data = {}
        x_labels = []

        for n_classes in COARSE_CFGS:
            sm = get_subject_scores(nd, region, args.folder, n_classes,
                                    "spearman", epoch=20)
            if len(sm) > 0:
                data[str(n_classes)] = sm
                x_labels.append(str(n_classes))

        sm_1k = get_subject_scores(nd, region, "imagenet1k", 1000,
                                   "spearman", epoch=20)
        if len(sm_1k) > 0:
            data["1K"] = sm_1k
            x_labels.append("1K")

        if len(x_labels) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            continue

        # Common subjects
        common = sorted(set.intersection(*(set(data[l].index) for l in x_labels)))

        # X positions with gap before 1K
        n_coarse = sum(1 for l in x_labels if l != "1K")
        x_pos = []
        for label in x_labels:
            x_pos.append(n_coarse + 0.7 if label == "1K" else len(x_pos))
        x_pos = np.array(x_pos, dtype=float)

        colors = []
        for label in x_labels:
            if label == "1K":
                colors.append("#7f7f7f")
            else:
                colors.append(blue_shades[COARSE_CFGS.index(int(label))])

        box_data = [data[l].loc[common].values for l in x_labels]

        bp = ax.boxplot(box_data, positions=x_pos, patch_artist=True, widths=0.5,
                        boxprops=dict(linewidth=1.0),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0),
                        medianprops=dict(linewidth=1.5, color="black"),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")

        # Connecting lines
        for subj in common:
            y_vals = [data[l].loc[subj] for l in x_labels]
            ax.plot(x_pos, y_vals, color="gray", alpha=0.25, linewidth=0.8, zorder=1)

        # Subject dots
        rng = np.random.default_rng(42)
        for i, label in enumerate(x_labels):
            y = data[label].loc[common].values
            xj = rng.normal(x_pos[i], 0.06, size=len(y))
            ax.scatter(xj, y, s=25, c="white", edgecolors="black",
                       linewidths=0.7, zorder=3, alpha=0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontweight="bold")
        ax.set_xlabel("Number of Classes", fontsize=13)
        ax.set_ylabel("Spearman \u03c1", fontsize=13)
        region_label = dcfg["region_labels"].get(region, region)
        ax.set_title(region_label, fontsize=15, fontweight="bold")

        all_vals = np.concatenate(box_data)
        yr = all_vals.max() - all_vals.min()
        ax.set_ylim(all_vals.min() - yr * 0.05, all_vals.max() + yr * 0.15)
        ax.yaxis.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, x_pos[-1] + 0.5)

        sns.despine(ax=ax, right=True, top=True, offset=5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

    fig.suptitle(
        f"Per-Subject Brain Alignment\n"
        f"({display_name}-PCA Labels, {args.dataset.upper()} RSA)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=1.0)
    out = f"plotters/figures/{args.dataset}/per_subject_{display_name.lower()}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coarseness progression: bars + per-subject")
    parser.add_argument("--dataset", required=True,
                        choices=["nsd", "tvsd", "things"])
    parser.add_argument("--folder", required=True,
                        choices=["pca_labels_alexnet", "pca_labels_vit",
                                 "pca_labels_clip", "pca_labels_dino"])
    args = parser.parse_args()

    dcfg = DATASET_CONFIG[args.dataset]
    plot_coarseness_bars(args, dcfg)
    plot_per_subject(args, dcfg)
