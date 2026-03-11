"""Alternative visualizations of coarseness data (brain alignment vs. number of classes).

Generates 5 plot types for both NSD and TVSD datasets:
1. Connected line plot with shaded CI
2. Heatmap
3. Slope/bump chart (rank trajectories)
4. Small multiples
5. Grouped bar plot with FancyBboxPatch

Run from project root:
    python manuscript/figures/misc/alt_coarseness.py
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import get_condition_summary, query_best_scores

# ── Constants ────────────────────────────────────────────────────────────────

COARSE_CFGS = [2, 4, 8, 16, 32, 64]

ARCHITECTURES = [
    ("alexnet", "pca_labels_alexnet", "AlexNet"),
    ("clip", "pca_labels_clip", "CLIP"),
    ("vit", "pca_labels_vit", "ViT"),
    ("pixels", "pca_labels_pixels", "Pixels"),
]

ARCH_STYLE = {
    "alexnet": {"color": "#2166AC", "marker": "o"},
    "clip":    {"color": "#1B7837", "marker": "s"},
    "vit":     {"color": "#C51B7D", "marker": "^"},
    "pixels":  {"color": "#E08214", "marker": "v"},
}

BASELINE_1K_COLOR = "#404040"

DATASET_CONFIGS = {
    "nsd": {
        "regions": ["early visual stream", "ventral visual stream"],
        "region_labels": {
            "early visual stream": "Early Visual Stream",
            "ventral visual stream": "Ventral Visual Stream",
        },
        "label": "NSD",
    },
    "tvsd": {
        "regions": ["V1", "IT"],
        "region_labels": {"V1": "V1", "IT": "IT"},
        "label": "TVSD",
    },
}

OUTPUT_DIR = "manuscript/figures/misc"


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_nsd_data(region, pca_folder, cfg_id, epoch=20):
    """Fetch NSD data using get_condition_summary (has bootstrap CIs)."""
    s = get_condition_summary("nsd", region, pca_folder, cfg_id,
                              "spearman", epoch=epoch, analysis="rsa")
    return s["mean"], s["ci_low"], s["ci_high"]


def fetch_tvsd_data(region, pca_folder, cfg_id, epoch=20):
    """Fetch TVSD data using query_best_scores (SEM from seeds)."""
    df = query_best_scores("tvsd", region, pca_folder, cfg_id,
                           "spearman", epoch=epoch, analysis="rsa")
    if df.empty:
        return np.nan, np.nan, np.nan
    seed_means = df.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
    ci_low = mean - 1.96 * sem
    ci_high = mean + 1.96 * sem
    return mean, ci_low, ci_high


def fetch_data(dataset, region, pca_folder, cfg_id, epoch=20):
    """Dispatch to the correct fetcher based on dataset."""
    if dataset == "nsd":
        return fetch_nsd_data(region, pca_folder, cfg_id, epoch=epoch)
    else:
        return fetch_tvsd_data(region, pca_folder, cfg_id, epoch=epoch)


def collect_all_data(dataset):
    """Collect data for all architectures, regions, and granularity levels.

    Returns a nested dict:
        data[region][arch_key] = {
            "means": [...],  "ci_low": [...],  "ci_high": [...],
            "cfgs": [2, 4, 8, ..., 64]
        }
        baselines[region] = {"mean": float, "ci_low": float, "ci_high": float}
    """
    dcfg = DATASET_CONFIGS[dataset]
    data = {}
    baselines = {}

    for region in dcfg["regions"]:
        data[region] = {}

        # Baseline (1K)
        bl_mean, bl_ci_lo, bl_ci_hi = fetch_data(
            dataset, region, "imagenet1k", 1000, epoch=20)
        baselines[region] = {"mean": bl_mean, "ci_low": bl_ci_lo, "ci_high": bl_ci_hi}

        for arch_key, pca_folder, display_name in ARCHITECTURES:
            means, ci_lows, ci_highs = [], [], []
            for cfg in COARSE_CFGS:
                m, cl, ch = fetch_data(dataset, region, pca_folder, cfg, epoch=20)
                means.append(m)
                ci_lows.append(cl)
                ci_highs.append(ch)

            data[region][arch_key] = {
                "means": np.array(means),
                "ci_low": np.array(ci_lows),
                "ci_high": np.array(ci_highs),
                "cfgs": COARSE_CFGS,
            }

    return data, baselines


# ── Plot 1: Connected line plot with shaded CI ──────────────────────────────

def plot_lines(dataset):
    """Connected line plot with shaded CI regions."""
    dcfg = DATASET_CONFIGS[dataset]
    data, baselines = collect_all_data(dataset)

    n_regions = len(dcfg["regions"])
    fig, axes = plt.subplots(1, n_regions, figsize=(5.5 * n_regions, 4.5),
                             sharey=False, squeeze=False)

    for idx, region in enumerate(dcfg["regions"]):
        ax = axes[0, idx]
        x = np.array(COARSE_CFGS)

        for arch_key, _, display_name in ARCHITECTURES:
            d = data[region][arch_key]
            style = ARCH_STYLE[arch_key]
            valid = ~np.isnan(d["means"])
            if not valid.any():
                continue

            xv = x[valid]
            mv = d["means"][valid]
            cl = d["ci_low"][valid]
            ch = d["ci_high"][valid]

            ax.plot(xv, mv, color=style["color"], marker=style["marker"],
                    markersize=6, linewidth=1.8, label=display_name, zorder=3)
            # Only shade if CIs are valid
            ci_valid = ~np.isnan(cl) & ~np.isnan(ch)
            if ci_valid.any():
                ax.fill_between(xv[ci_valid], cl[ci_valid], ch[ci_valid],
                                color=style["color"], alpha=0.15, zorder=1)

        # Baseline
        bl = baselines[region]
        if not np.isnan(bl["mean"]):
            ax.axhline(bl["mean"], color=BASELINE_1K_COLOR, linestyle="--",
                       linewidth=1.8, label="1K Baseline", zorder=2)
            if not np.isnan(bl["ci_low"]) and not np.isnan(bl["ci_high"]):
                ax.axhspan(bl["ci_low"], bl["ci_high"],
                           color=BASELINE_1K_COLOR, alpha=0.08, zorder=0)

        ax.set_xscale("log", base=2)
        ax.set_xticks(COARSE_CFGS)
        ax.set_xticklabels([str(c) for c in COARSE_CFGS])
        ax.set_xlabel("Number of Classes", fontsize=11)
        ax.set_ylabel("Spearman $\\rho$", fontsize=11)
        ax.set_title(dcfg["region_labels"][region], fontsize=13, fontweight="bold")
        sns.despine(ax=ax, right=True, top=True, offset=5)

        if idx == n_regions - 1:
            ax.legend(fontsize=8.5, frameon=True, framealpha=0.9,
                      edgecolor="gray", loc="best")

    fig.suptitle(f"Brain Alignment vs. Label Granularity ({dcfg['label']})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    out = os.path.join(OUTPUT_DIR, f"alt_coarseness_lines_{dataset}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Plot 2: Heatmap ─────────────────────────────────────────────────────────

def plot_heatmap(dataset):
    """Heatmap: architectures x granularity, color = score."""
    dcfg = DATASET_CONFIGS[dataset]
    data, baselines = collect_all_data(dataset)

    n_regions = len(dcfg["regions"])
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions + 1, 3.2),
                             squeeze=False)

    arch_labels = [disp for _, _, disp in ARCHITECTURES]
    arch_keys = [k for k, _, _ in ARCHITECTURES]

    for idx, region in enumerate(dcfg["regions"]):
        ax = axes[0, idx]

        matrix = np.full((len(ARCHITECTURES), len(COARSE_CFGS)), np.nan)
        for ai, arch_key in enumerate(arch_keys):
            d = data[region][arch_key]
            matrix[ai, :] = d["means"]

        sns.heatmap(matrix, ax=ax, cmap="magma", annot=True, fmt=".3f",
                    annot_kws={"fontsize": 8}, linewidths=0.5, linecolor="white",
                    xticklabels=[str(c) for c in COARSE_CFGS],
                    yticklabels=arch_labels if idx == 0 else False,
                    cbar=idx == n_regions - 1,
                    cbar_kws={"label": "Spearman $\\rho$", "shrink": 0.8})

        ax.set_xlabel("Number of Classes", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Architecture", fontsize=11)
        ax.set_title(dcfg["region_labels"][region], fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(f"Alignment Heatmap ({dcfg['label']})",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, f"alt_coarseness_heatmap_{dataset}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Plot 3: Slope/bump chart ────────────────────────────────────────────────

def plot_bump(dataset):
    """Bump chart: rank trajectories across granularity levels."""
    dcfg = DATASET_CONFIGS[dataset]
    data, baselines = collect_all_data(dataset)

    n_regions = len(dcfg["regions"])
    fig, axes = plt.subplots(1, n_regions, figsize=(5.5 * n_regions, 4.5),
                             sharey=True, squeeze=False)

    arch_keys = [k for k, _, _ in ARCHITECTURES]

    for idx, region in enumerate(dcfg["regions"]):
        ax = axes[0, idx]

        # Build score matrix: architectures x granularity
        score_matrix = np.full((len(ARCHITECTURES), len(COARSE_CFGS)), np.nan)
        for ai, arch_key in enumerate(arch_keys):
            score_matrix[ai, :] = data[region][arch_key]["means"]

        # Compute ranks at each granularity (1 = best = highest score)
        rank_matrix = np.full_like(score_matrix, np.nan)
        for ci in range(len(COARSE_CFGS)):
            col = score_matrix[:, ci]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() > 0:
                # Rank descending: highest score gets rank 1
                valid_scores = col[valid_mask]
                temp_ranks = len(valid_scores) - np.argsort(np.argsort(valid_scores))
                rank_matrix[valid_mask, ci] = temp_ranks

        x = np.arange(len(COARSE_CFGS))
        for ai, (arch_key, _, display_name) in enumerate(ARCHITECTURES):
            style = ARCH_STYLE[arch_key]
            ranks = rank_matrix[ai, :]
            valid = ~np.isnan(ranks)
            if not valid.any():
                continue

            ax.plot(x[valid], ranks[valid], color=style["color"],
                    marker=style["marker"], markersize=8, linewidth=2.2,
                    label=display_name, zorder=3)

            # Label endpoints
            if valid.any():
                first_valid = np.where(valid)[0][0]
                last_valid = np.where(valid)[0][-1]
                ax.annotate(display_name,
                            xy=(x[last_valid] + 0.15, ranks[last_valid]),
                            fontsize=8.5, color=style["color"], fontweight="bold",
                            va="center")

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in COARSE_CFGS])
        ax.set_xlabel("Number of Classes", fontsize=11)
        ax.set_ylabel("Rank (1 = Best)", fontsize=11)
        ax.set_title(dcfg["region_labels"][region], fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.set_yticks(range(1, len(ARCHITECTURES) + 1))
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, right=True, top=True, offset=5)

    fig.suptitle(f"Architecture Ranking Across Granularity ({dcfg['label']})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2)
    out = os.path.join(OUTPUT_DIR, f"alt_coarseness_bump_{dataset}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Plot 4: Small multiples ─────────────────────────────────────────────────

def plot_multiples(dataset):
    """Small multiples: one mini-panel per architecture, rows = regions."""
    dcfg = DATASET_CONFIGS[dataset]
    data, baselines = collect_all_data(dataset)

    n_regions = len(dcfg["regions"])
    n_archs = len(ARCHITECTURES)
    fig, axes = plt.subplots(n_regions, n_archs,
                             figsize=(3.5 * n_archs, 3.5 * n_regions),
                             sharey="row", squeeze=False)

    x = np.array(COARSE_CFGS)

    for ri, region in enumerate(dcfg["regions"]):
        for ai, (arch_key, _, display_name) in enumerate(ARCHITECTURES):
            ax = axes[ri, ai]
            d = data[region][arch_key]
            style = ARCH_STYLE[arch_key]
            valid = ~np.isnan(d["means"])

            if valid.any():
                xv = x[valid]
                mv = d["means"][valid]
                cl = d["ci_low"][valid]
                ch = d["ci_high"][valid]

                ax.plot(xv, mv, color=style["color"], marker=style["marker"],
                        markersize=5, linewidth=1.5, zorder=3)

                ci_valid = ~np.isnan(cl) & ~np.isnan(ch)
                if ci_valid.any():
                    ax.fill_between(xv[ci_valid], cl[ci_valid], ch[ci_valid],
                                    color=style["color"], alpha=0.2, zorder=1)

            # Baseline
            bl = baselines[region]
            if not np.isnan(bl["mean"]):
                ax.axhline(bl["mean"], color=BASELINE_1K_COLOR, linestyle="--",
                           linewidth=1.2, zorder=2, alpha=0.7)

            ax.set_xscale("log", base=2)
            ax.set_xticks(COARSE_CFGS)
            ax.set_xticklabels([str(c) for c in COARSE_CFGS], fontsize=7)
            ax.tick_params(axis="both", labelsize=8)

            if ri == n_regions - 1:
                ax.set_xlabel("Classes", fontsize=9)
            if ai == 0:
                ax.set_ylabel("Spearman $\\rho$", fontsize=9)
            if ri == 0:
                ax.set_title(display_name, fontsize=11, fontweight="bold",
                             color=style["color"])

            # Region label on leftmost panel
            if ai == 0:
                ax.annotate(dcfg["region_labels"][region], xy=(-0.45, 0.5),
                            xycoords="axes fraction", fontsize=10, fontweight="bold",
                            rotation=90, ha="center", va="center")

            sns.despine(ax=ax, right=True, top=True, offset=3)

    fig.suptitle(f"Per-Architecture Alignment Trends ({dcfg['label']})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    out = os.path.join(OUTPUT_DIR, f"alt_coarseness_multiples_{dataset}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Plot 5: Grouped bar plot ────────────────────────────────────────────────

def plot_bars(dataset):
    """Grouped bar plot with FancyBboxPatch and 1K baseline dashed line."""
    dcfg = DATASET_CONFIGS[dataset]
    data, baselines = collect_all_data(dataset)

    n_regions = len(dcfg["regions"])
    n_archs = len(ARCHITECTURES)
    fig, axes = plt.subplots(1, n_regions, figsize=(6 * n_regions, 4.5),
                             sharey=False, squeeze=False)

    bar_width = 0.18
    group_offsets = np.arange(n_archs) - (n_archs - 1) / 2

    for idx, region in enumerate(dcfg["regions"]):
        ax = axes[0, idx]
        x_base = np.arange(len(COARSE_CFGS))

        for ai, (arch_key, _, display_name) in enumerate(ARCHITECTURES):
            d = data[region][arch_key]
            style = ARCH_STYLE[arch_key]

            for ci, cfg in enumerate(COARSE_CFGS):
                mean_val = d["means"][ci]
                if np.isnan(mean_val):
                    continue

                x_pos = x_base[ci] + group_offsets[ai] * bar_width
                # FancyBboxPatch with rounded corners
                rect = mpatches.FancyBboxPatch(
                    (x_pos - bar_width / 2, 0), bar_width, mean_val,
                    boxstyle=mpatches.BoxStyle("Round", pad=0.01,
                                               rounding_size=0.04),
                    facecolor=style["color"], edgecolor="black",
                    linewidth=0.7, mutation_aspect=0.04, zorder=3,
                    alpha=0.85,
                )
                ax.add_patch(rect)

                # Error bar
                ci_lo = d["ci_low"][ci]
                ci_hi = d["ci_high"][ci]
                if not np.isnan(ci_lo) and not np.isnan(ci_hi):
                    err_lo = max(0, mean_val - ci_lo)
                    err_hi = max(0, ci_hi - mean_val)
                    if err_lo > 0 or err_hi > 0:
                        ax.errorbar(x_pos, mean_val,
                                    yerr=[[err_lo], [err_hi]],
                                    fmt="none", ecolor="black",
                                    elinewidth=0.8, capsize=2, capthick=0.8,
                                    zorder=5)

        # Baseline 1K line
        bl = baselines[region]
        if not np.isnan(bl["mean"]):
            ax.axhline(bl["mean"], color=BASELINE_1K_COLOR, linestyle="--",
                       linewidth=1.8, label="1K Baseline", zorder=2, alpha=0.9)

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(c) for c in COARSE_CFGS])
        ax.set_xlabel("Number of Classes", fontsize=11)
        ax.set_ylabel("Spearman $\\rho$", fontsize=11)
        ax.set_title(dcfg["region_labels"][region], fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

        # Auto y-limits
        all_vals = []
        for arch_key in [k for k, _, _ in ARCHITECTURES]:
            vals = data[region][arch_key]["means"]
            all_vals.extend(vals[~np.isnan(vals)])
        if bl["mean"] is not None and not np.isnan(bl["mean"]):
            all_vals.append(bl["mean"])
        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals)
            margin = (y_max - y_min) * 0.15 if y_max > y_min else 0.01
            ax.set_ylim(max(0, y_min - margin), y_max + margin)
        ax.autoscale_view()

        sns.despine(ax=ax, right=True, top=True, offset=5)

        # Legend on last subplot
        if idx == n_regions - 1:
            handles = []
            for arch_key, _, display_name in ARCHITECTURES:
                style = ARCH_STYLE[arch_key]
                handles.append(mpatches.Patch(facecolor=style["color"],
                                              edgecolor="black", linewidth=0.7,
                                              label=display_name, alpha=0.85))
            handles.append(plt.Line2D([], [], color=BASELINE_1K_COLOR,
                                      linestyle="--", linewidth=1.8,
                                      label="1K Baseline"))
            ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9,
                      edgecolor="gray", loc="best")

    fig.suptitle(f"Grouped Bar: Alignment by Granularity ({dcfg['label']})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.0)
    out = os.path.join(OUTPUT_DIR, f"alt_coarseness_bars_{dataset}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset in ["nsd", "tvsd"]:
        print(f"\n{'='*60}")
        print(f"Generating plots for {dataset.upper()}")
        print(f"{'='*60}")

        plot_lines(dataset)
        plot_heatmap(dataset)
        plot_bump(dataset)
        plot_multiples(dataset)
        plot_bars(dataset)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
