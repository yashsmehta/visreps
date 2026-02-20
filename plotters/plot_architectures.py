"""Unified architecture comparison plotter.

Generates two figures for a given dataset and region:
  1. Grouped bars — all architectures across coarseness levels, 1K baseline line
  2. Per-subject boxes — architectures at their best coarse cfg, with subject dots

Auto-discovers which architectures have data in results.db.

Usage:
    python plotters/plot_architectures.py --dataset nsd --region "ventral visual stream"
    python plotters/plot_architectures.py --dataset tvsd --region V4
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import seaborn as sns

sys.path.insert(0, "plotters")
from plotter_utils import query_best_scores, get_subject_scores

# ── Configuration ─────────────────────────────────────────────────────────
KNOWN_FOLDERS = [
    ("alexnet", "pca_labels_alexnet"),
    ("vit", "pca_labels_vit"),
    ("dino", "pca_labels_dino"),
    ("clip", "pca_labels_clip"),
]

NEURAL_DATASET_MAP = {
    "nsd": "nsd",
    "tvsd": "tvsd",
    "things": "things-behavior",
}

COARSE_CFGS = [2, 4, 8, 16, 32, 64]

COLOR_MAP = {
    'alexnet': '#1f77b4',
    'vit': '#ee854a',
    'dino': '#ff7f0e',
    'clip': '#2d7f2d',
}

NAME_MAP = {
    'alexnet': 'AlexNet',
    'vit': 'ViT',
    'dino': 'DINO',
    'clip': 'CLIP',
}

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.1)


def discover_architectures(nd, region):
    """Find which architectures have data for this dataset+region."""
    available = []
    for arch_key, folder in KNOWN_FOLDERS:
        for cfg in COARSE_CFGS:
            df = query_best_scores(nd, region, folder, cfg, "spearman", epoch=20)
            if not df.empty:
                available.append((arch_key, folder))
                break
    return available


# ── Figure 1: Grouped bars across coarseness levels ──────────────────────
def plot_architecture_bars(args, nd, region, available_archs):
    """Grouped bar chart: each architecture at each coarseness level + 1K line."""
    n_archs = len(available_archs)
    bar_width = 0.24
    intra_gap = 0.04
    group_gap = 0.30

    scores_by_arch_class = {}

    # Coarse-grained bars
    for arch_key, folder in available_archs:
        for cfg in COARSE_CFGS:
            df = query_best_scores(nd, region, folder, cfg, "spearman", epoch=20)
            if not df.empty:
                scores_by_arch_class[(arch_key, cfg)] = df["score"].tolist()

    # 1K baseline
    df_1k = query_best_scores(nd, region, "imagenet1k", 1000, "spearman", epoch=20)
    scores_1k = df_1k["score"].tolist() if not df_1k.empty else None

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(COARSE_CFGS)), 5))

    # Draw grouped bars with FancyBboxPatch
    for i, cfg in enumerate(COARSE_CFGS):
        base = i * (n_archs * bar_width + (n_archs - 1) * intra_gap + group_gap)
        for j, (arch_key, _) in enumerate(available_archs):
            if (arch_key, cfg) in scores_by_arch_class:
                scores = scores_by_arch_class[(arch_key, cfg)]
                mean_val = np.mean(scores)
                bar_pos = base + j * (bar_width + intra_gap)
                rect = mpatches.FancyBboxPatch(
                    (bar_pos, 0), bar_width, mean_val,
                    boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
                    facecolor=COLOR_MAP[arch_key], edgecolor="black",
                    linewidth=1.0, mutation_aspect=0.05,
                )
                ax.add_patch(rect)

    # 1K baseline line
    if scores_1k:
        mean_1k = np.mean(scores_1k)
        ax.axhline(y=mean_1k, color="#666666", linestyle="--", linewidth=2.5,
                   label="ImageNet-1K", zorder=2, alpha=0.9)

    # X-axis
    tick_pos, tick_labels = [], []
    for i, cfg in enumerate(COARSE_CFGS):
        base = i * (n_archs * bar_width + (n_archs - 1) * intra_gap + group_gap)
        group_w = n_archs * bar_width + (n_archs - 1) * intra_gap
        tick_pos.append(base + group_w / 2)
        tick_labels.append(str(cfg))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontweight="bold")
    ax.tick_params(axis="x", direction="out", length=5, width=1.5, pad=8, labelsize=14)

    # Y-axis
    ax.tick_params(axis="y", which="major", direction="out", labelsize=13,
                   length=6, width=1.5, pad=6)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: "" if np.isclose(x, 0) else f"{x:.2f}"))
    ax.tick_params(axis="y", which="minor", direction="out", length=3, width=1.0)

    # Limits
    all_means = [np.mean(v) for v in scores_by_arch_class.values()]
    if scores_1k:
        all_means.append(np.mean(scores_1k))
    y_max = max(all_means) if all_means else 0.1
    ax.set_ylim(0, y_max + 0.025)

    max_pos = (len(COARSE_CFGS) - 1) * (n_archs * bar_width + (n_archs - 1) * intra_gap + group_gap)
    max_pos += n_archs * bar_width + (n_archs - 1) * intra_gap + 0.5
    ax.set_xlim(-0.5, max_pos)

    ax.set_ylabel("Spearman \u03c1", fontsize=15, labelpad=12)
    region_title = region.title() if len(region) < 5 else region.replace("visual stream", "Visual Stream").title()
    ax.set_title(f"Architecture Comparison — {region_title}",
                 fontsize=16, fontweight="bold", pad=15)

    # Legend
    handles = [mpatches.Patch(facecolor=COLOR_MAP[ak], edgecolor="black",
               linewidth=1.0, label=f"{NAME_MAP[ak]} PCA")
               for ak, _ in available_archs]
    if scores_1k:
        handles.append(mlines.Line2D([], [], color="#666666", linestyle="--",
                                     linewidth=2.5, label="ImageNet-1K"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5),
              frameon=True, fontsize=13, framealpha=0.95, edgecolor="black")

    sns.despine(right=True, top=True, offset=8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_linewidth(1.8)

    plt.tight_layout(pad=1.2, rect=[0, 0, 0.85, 1])
    region_slug = region.lower().replace(" ", "_")
    out = f"plotters/figures/{args.dataset}/arch_bars_{region_slug}.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved -> {out}")


# ── Figure 2: Per-subject boxes at best cfg per architecture ─────────────
def plot_architecture_boxes(args, nd, region, available_archs):
    """Box plots comparing architectures at their best coarse cfg."""
    data_dict = {}
    labels = []

    # 1K baseline
    scores_1k = get_subject_scores(nd, region, "imagenet1k", 1000,
                                   "spearman", epoch=20)
    if len(scores_1k) > 0:
        data_dict["1K"] = scores_1k.values
        labels.append("1K")
        print(f"  1K: {len(scores_1k)} subjects, mean={scores_1k.mean():.4f}")

    # Each architecture at its best cfg
    for arch_key, folder in available_archs:
        best_cfg, best_mean = None, -np.inf
        for cfg in COARSE_CFGS:
            s = get_subject_scores(nd, region, folder, cfg, "spearman", epoch=20)
            if len(s) > 0 and s.mean() > best_mean:
                best_mean = s.mean()
                best_cfg = cfg
        if best_cfg is not None:
            s = get_subject_scores(nd, region, folder, best_cfg,
                                   "spearman", epoch=20)
            label = f"{NAME_MAP[arch_key]} ({best_cfg})"
            data_dict[label] = s.values
            labels.append(label)
            print(f"  {label}: {len(s)} subjects, mean={s.mean():.4f}")

    if len(labels) < 2:
        print("Not enough data for architecture box plot")
        return

    # Colors: grey for 1K, then architecture colors
    arch_colors = [COLOR_MAP[ak] for ak, _ in available_archs]
    colors = (["#7f7f7f"] if "1K" in labels else []) + arch_colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5))
    box_data = [data_dict[l] for l in labels]

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color="black"),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for i, label in enumerate(labels):
        y = data_dict[label]
        x = rng.normal(i + 1, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.8, s=35, c="white", edgecolors="black",
                   linewidths=0.8, zorder=3)

    all_vals = np.concatenate(box_data)
    yr = all_vals.max() - all_vals.min()
    y_bot = np.floor(all_vals.min() * 20) / 20
    y_top = np.ceil((all_vals.max() + yr * 0.1) * 20) / 20
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks(np.arange(y_bot, y_top + 0.01, 0.05))

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11)
    ax.set_ylabel("Spearman \u03c1", fontsize=13)
    ax.set_xlabel("PCA Label Source", fontsize=13)

    region_title = region.title() if len(region) < 5 else region.replace("visual stream", "Visual Stream").title()
    ax.set_title(f"Architecture Comparison — {region_title}\n(Best Coarse per Arch)",
                 fontsize=14, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    region_slug = region.lower().replace(" ", "_")
    out = f"plotters/figures/{args.dataset}/arch_boxes_{region_slug}.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved -> {out}")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Architecture comparison: grouped bars + per-subject boxes")
    parser.add_argument("--dataset", required=True, choices=["nsd", "tvsd", "things"])
    parser.add_argument("--region", required=True,
                        help="Brain region (e.g. 'ventral visual stream', 'V4', 'N/A')")
    args = parser.parse_args()

    nd = NEURAL_DATASET_MAP[args.dataset]
    region = args.region

    print(f"Discovering architectures for {args.dataset} / {region}...")
    available = discover_architectures(nd, region)
    if not available:
        print("No architecture data found")
        sys.exit(1)

    print(f"Found: {[NAME_MAP[a] for a, _ in available]}")
    plot_architecture_bars(args, nd, region, available)
    plot_architecture_boxes(args, nd, region, available)
