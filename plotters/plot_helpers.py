"""Shared coarseness plotting logic.

Provides the core drawing functions for coarseness bar plots and per-subject
box plots.  Per-dataset scripts import from here and supply only a config
dict, a PCA-folder name, and an output directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

from plotter_utils import get_condition_summary, get_subject_scores

# ── Constants ────────────────────────────────────────────────────────────
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
N_COARSE = len(COARSE_CFGS)

FOLDER_DISPLAY = {
    "pca_labels_alexnet": "AlexNet",
    "pca_labels_vit": "ViT",
    "pca_labels_clip": "CLIP",
    "pca_labels_dino": "DINO",
}

# ── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams["hatch.color"] = "grey"

blues = sns.color_palette("Blues", n_colors=N_COARSE + 1)[1:]
UNTRAINED_COLOR = "#AAAAAA"
BASELINE_COLOR = "#FFA500"
BAR_WIDTH = 0.72


# ── Layout helper ────────────────────────────────────────────────────────
def make_figure(dcfg):
    """Create figure + ordered axes list, respecting optional 'layout' key."""
    n_regions = len(dcfg["regions"])
    layout = dcfg.get("layout")
    if layout:
        nrows, ncols, positions = layout
        scale = 1 + 0.25 * (ncols - 1)
        fig, grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                 sharey=False, squeeze=False)
        ax_list = [grid[r, c] for r, c in positions]
        used = set(positions)
        for r in range(nrows):
            for c in range(ncols):
                if (r, c) not in used:
                    grid[r, c].set_visible(False)
    else:
        scale = 1 + 0.35 * (n_regions - 1)
        fig, grid = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4 * scale),
                                 sharey=False, squeeze=False)
        ax_list = [grid[0, i] for i in range(n_regions)]
    return fig, ax_list, scale


# ── Fancy bar helpers ────────────────────────────────────────────────────
def draw_fancy_bar(ax, x, height, color, hatch="", width=BAR_WIDTH, scale=1.0):
    """Draw a clean bar with optional hatching."""
    ax.bar(x, height, width=width, color=color, edgecolor="black",
           linewidth=0.8 * scale, hatch=hatch, zorder=3)


def draw_break_marks(ax, x, scale=1.0):
    """Draw diagonal slashes at the bottom spine to indicate axis break."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    spine_y, dy, dx, gap = -0.022, 0.028, 0.20, 0.13
    ax.plot([x - gap - dx - 0.1, x + gap + dx + 0.1], [spine_y, spine_y],
            color="white", linewidth=5 * scale, transform=trans, clip_on=False, zorder=9)
    for offset in [-gap, gap]:
        ax.plot([x + offset - dx, x + offset + dx], [spine_y - dy, spine_y + dy],
                color="black", linewidth=1.8 * scale, clip_on=False, zorder=10,
                transform=trans)


# ── Figure 1: Coarseness bars ───────────────────────────────────────────
def plot_coarseness_bars(dcfg, folder, output_dir, dataset_label=None):
    """Fancy bar plot: untrained | coarse (2-64) | break | 1000.

    Parameters
    ----------
    dcfg : dict
        Dataset config with keys: neural_dataset, regions, region_labels, has_subjects,
        and optionally layout, output_suffix.
    folder : str
        PCA labels folder name (e.g. "pca_labels_alexnet").
    output_dir : str
        Directory to save the figure into (e.g. "plotters/nsd/figures").
    dataset_label : str, optional
        Label for the suptitle (e.g. "NSD"). Defaults to neural_dataset uppercased.
    """
    nd = dcfg["neural_dataset"]
    regions = dcfg["regions"]
    compare_method = dcfg.get("compare_method", "spearman")
    analysis_label = "Encoding Score" if dcfg.get("analysis", "rsa") == "encoding_score" else "RSA"
    y_label = "Pearson r" if compare_method == "pearson" else "Spearman \u03c1"
    display_name = FOLDER_DISPLAY.get(folder, folder)
    if dataset_label is None:
        dataset_label = nd.upper()

    fig, ax_list, scale = make_figure(dcfg)

    for idx, region in enumerate(regions):
        ax = ax_list[idx]

        # Check for untrained data (epoch=0)
        un = get_condition_summary(nd, region, "imagenet1k", 1000,
                                   compare_method, epoch=0)
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
            s = get_condition_summary(nd, region, folder, cfg,
                                      compare_method, epoch=20)
            all_means.append(s["mean"])
            all_ci_lo.append(s["ci_low"])
            all_ci_hi.append(s["ci_high"])
            all_x.append(X_COARSE[i])
            all_colors.append(blues[i])
            all_hatches.append("/")
            all_labels.append(str(cfg))

        bl = get_condition_summary(nd, region, "imagenet1k", 1000,
                                   compare_method, epoch=20)
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
                               all_colors[k], all_hatches[k], scale=scale)

        # Error bars (both must be non-negative for matplotlib)
        for k in range(len(all_x)):
            if (not np.isnan(err_lo[k]) and not np.isnan(err_hi[k])
                    and err_lo[k] >= 0 and err_hi[k] >= 0
                    and (err_lo[k] > 0 or err_hi[k] > 0)):
                ax.errorbar(all_x[k], all_means[k],
                            yerr=[[err_lo[k]], [err_hi[k]]],
                            fmt="none", ecolor="black", elinewidth=1.0 * scale,
                            capsize=4 * scale, capthick=1.0 * scale, zorder=5)

        draw_break_marks(ax, BREAK_X, scale=scale)

        # Axis formatting
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels, fontsize=10 * scale, ha="center")
        ax.tick_params(axis="x", direction="out", bottom=False,
                       length=4 * scale, width=1.5 * scale)
        ax.tick_params(axis="y", which="major", direction="out",
                       labelsize=12 * scale, length=5 * scale, width=1.5 * scale)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", direction="out",
                       length=3 * scale, width=1.0 * scale)
        first_x = X_UNTRAINED if has_untrained else all_x[0]
        ax.set_xlim(first_x - 0.6, X_BASELINE + 0.7)
        ax.set_ylim(y_bottom, y_top)

        ax.set_xlabel("Number of Classes", fontsize=13 * scale)
        ax.set_ylabel(y_label, fontsize=13 * scale)
        region_label = dcfg["region_labels"].get(region, region)
        ax.set_title(region_label, fontsize=15 * scale, fontweight="bold",
                     pad=10 * scale)

        sns.despine(ax=ax, right=True, top=True, offset=5 * scale)
        ax.spines["bottom"].set_linewidth(1.5 * scale)
        ax.spines["left"].set_linewidth(1.5 * scale)

    fig.suptitle(
        f"Brain Alignment Across Label Granularity\n"
        f"({display_name}-PCA Labels, {dataset_label} {analysis_label})",
        fontsize=16 * scale, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=1.0)
    suffix = dcfg.get("output_suffix", "")
    out = f"{output_dir}/coarseness_bars_{display_name.lower()}{suffix}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


# ── Figure 2: Per-subject boxes ─────────────────────────────────────────
def plot_per_subject(dcfg, folder, output_dir, dataset_label=None):
    """Box plots with per-subject dots connected across class counts.

    Parameters
    ----------
    dcfg : dict
        Dataset config (same as plot_coarseness_bars).
    folder : str
        PCA labels folder name.
    output_dir : str
        Directory to save the figure into.
    dataset_label : str, optional
        Label for the suptitle. Defaults to neural_dataset uppercased.
    """
    if not dcfg["has_subjects"]:
        print(f"Skipping per-subject plot ({dcfg['neural_dataset']} has no subjects)")
        return

    nd = dcfg["neural_dataset"]
    regions = dcfg["regions"]
    compare_method = dcfg.get("compare_method", "spearman")
    analysis_label = "Encoding Score" if dcfg.get("analysis", "rsa") == "encoding_score" else "RSA"
    y_label = "Pearson r" if compare_method == "pearson" else "Spearman \u03c1"
    display_name = FOLDER_DISPLAY.get(folder, folder)
    if dataset_label is None:
        dataset_label = nd.upper()

    fig, ax_list, scale = make_figure(dcfg)

    blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]

    for idx, region in enumerate(regions):
        ax = ax_list[idx]
        data = {}
        x_labels = []

        for n_classes in COARSE_CFGS:
            sm = get_subject_scores(nd, region, folder, n_classes,
                                    compare_method, epoch=20)
            if len(sm) > 0:
                data[str(n_classes)] = sm
                x_labels.append(str(n_classes))

        sm_1k = get_subject_scores(nd, region, "imagenet1k", 1000,
                                   compare_method, epoch=20)
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
                        boxprops=dict(linewidth=1.0 * scale),
                        whiskerprops=dict(linewidth=1.0 * scale),
                        capprops=dict(linewidth=1.0 * scale),
                        medianprops=dict(linewidth=1.5 * scale, color="black"),
                        flierprops=dict(marker="o", markersize=3 * scale, alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")

        # Connecting lines
        for subj in common:
            y_vals = [data[l].loc[subj] for l in x_labels]
            ax.plot(x_pos, y_vals, color="gray", alpha=0.25,
                    linewidth=0.8 * scale, zorder=1)

        # Subject dots
        rng = np.random.default_rng(42)
        for i, label in enumerate(x_labels):
            y = data[label].loc[common].values
            xj = rng.normal(x_pos[i], 0.06, size=len(y))
            ax.scatter(xj, y, s=25 * scale, c="white", edgecolors="black",
                       linewidths=0.7 * scale, zorder=3, alpha=0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontweight="bold", fontsize=11 * scale)
        ax.set_xlabel("Number of Classes", fontsize=13 * scale)
        ax.set_ylabel(y_label, fontsize=13 * scale)
        region_label = dcfg["region_labels"].get(region, region)
        ax.set_title(region_label, fontsize=15 * scale, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11 * scale)

        all_vals = np.concatenate(box_data)
        yr = all_vals.max() - all_vals.min()
        ax.set_ylim(all_vals.min() - yr * 0.05, all_vals.max() + yr * 0.15)
        ax.yaxis.grid(True, linestyle="-", alpha=0.3, linewidth=0.5 * scale)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, x_pos[-1] + 0.5)

        sns.despine(ax=ax, right=True, top=True, offset=5 * scale)
        ax.spines["bottom"].set_linewidth(1.5 * scale)
        ax.spines["left"].set_linewidth(1.5 * scale)

    fig.suptitle(
        f"Per-Subject Brain Alignment\n"
        f"({display_name}-PCA Labels, {dataset_label} {analysis_label})",
        fontsize=16 * scale, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=1.0)
    suffix = dcfg.get("output_suffix", "")
    out = f"{output_dir}/per_subject_{display_name.lower()}{suffix}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()
