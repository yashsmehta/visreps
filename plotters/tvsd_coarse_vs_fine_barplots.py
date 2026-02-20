"""
TVSD RSA: Coarse-grained vs fine-grained (1000-way) bar plots.

Plot 1: Best coarse-grained vs 1000-way per region (V1, V4, IT)
Plot 2: All granularities (2–64, 1000) per region, 3 subplots

Data pipeline:
  1. Average over 2 subjects per (n_classes, region, layer, seed)
  2. Pick best layer per (n_classes, region, seed)
  3. Report mean ± SEM across 3 seeds
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ── Nature-style rcParams ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'lines.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'figure.dpi': 150,
})

# ── Config ──
CSV_PATH = 'logs/all_rsa_tvsd.csv'
CHECKPOINT_DIRS = ['/data/ymehta3/alexnet_pca', '/data/ymehta3/default']
REGIONS = ['V1', 'V4', 'IT']
COARSE_COLOR = '#4878A8'
BASELINE_COLOR = '#888888'
POINT_COLOR = '#2C2C2C'


def load_and_summarize(csv_path, checkpoint_dirs):
    """Load CSV, filter to relevant checkpoint dirs, compute summary stats."""
    df = pd.read_csv(csv_path)
    df = df[df['checkpoint_dir'].isin(checkpoint_dirs)].copy()
    df['n_classes'] = df['cfg_id']

    # Avg over subjects → best layer per seed → mean ± SEM across seeds
    by_subj = df.groupby(['n_classes', 'region', 'layer', 'seed'])['score'].mean().reset_index()
    best_per_seed = by_subj.loc[by_subj.groupby(['n_classes', 'region', 'seed'])['score'].idxmax()]
    summary = best_per_seed.groupby(['n_classes', 'region']).agg(
        mean=('score', 'mean'),
        sem=('score', lambda x: x.std() / np.sqrt(len(x))),
        seeds=('score', list)
    ).reset_index()
    return summary


def plot_best_coarse_vs_1k(summary, regions, out_path):
    """Plot 1: Best coarse-grained vs 1000-way per region."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bar_width = 0.32
    x = np.arange(len(regions))

    coarse_means, coarse_sems, coarse_seeds_all = [], [], []
    baseline_means, baseline_sems, baseline_seeds_all = [], [], []
    best_n_classes = []

    for region in regions:
        reg_coarse = summary[(summary['region'] == region) & (summary['n_classes'] != 1000)]
        best_row = reg_coarse.loc[reg_coarse['mean'].idxmax()]
        coarse_means.append(best_row['mean'])
        coarse_sems.append(best_row['sem'])
        coarse_seeds_all.append(best_row['seeds'])
        best_n_classes.append(int(best_row['n_classes']))

        bl = summary[(summary['region'] == region) & (summary['n_classes'] == 1000)]
        baseline_means.append(bl['mean'].values[0])
        baseline_sems.append(bl['sem'].values[0])
        baseline_seeds_all.append(bl['seeds'].values[0])

    coarse_means = np.array(coarse_means)
    baseline_means = np.array(baseline_means)
    coarse_sems = np.array(coarse_sems)
    baseline_sems = np.array(baseline_sems)

    ax.bar(x - bar_width/2 - 0.02, coarse_means, bar_width,
           color=COARSE_COLOR, edgecolor='white', linewidth=0.3,
           yerr=coarse_sems, capsize=2, error_kw={'linewidth': 0.6, 'capthick': 0.6})
    ax.bar(x + bar_width/2 + 0.02, baseline_means, bar_width,
           color=BASELINE_COLOR, edgecolor='white', linewidth=0.3,
           yerr=baseline_sems, capsize=2, error_kw={'linewidth': 0.6, 'capthick': 0.6})

    np.random.seed(42)
    for i in range(len(regions)):
        jitter = np.random.uniform(-0.04, 0.04, size=3)
        ax.scatter(x[i] - bar_width/2 - 0.02 + jitter, coarse_seeds_all[i],
                   s=8, color=POINT_COLOR, zorder=5, alpha=0.7, linewidths=0.3, edgecolors='white')
        ax.scatter(x[i] + bar_width/2 + 0.02 + jitter, baseline_seeds_all[i],
                   s=8, color=POINT_COLOR, zorder=5, alpha=0.7, linewidths=0.3, edgecolors='white')

    for i, n in enumerate(best_n_classes):
        ax.text(x[i] - bar_width/2 - 0.02, -0.012, f'{n}CLS',
                ha='center', va='top', fontsize=5, color=COARSE_COLOR, fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontweight='bold')
    ax.set_ylabel('RSA Score', labelpad=4)
    ax.set_title('Best Coarse-Grained vs 1000-Way', fontweight='bold', pad=8)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_ylim(0, max(max(coarse_means), max(baseline_means)) + 0.04)

    legend_handles = [
        mpatches.Patch(facecolor=COARSE_COLOR, edgecolor='none', label='Best coarse-grained'),
        mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor='none', label='1000-way'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', frameon=False, handlelength=1.2, handletextpad=0.4)

    plt.tight_layout(pad=0.8)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot 1 saved: {out_path}")


def plot_all_granularities(summary, regions, out_path):
    """Plot 2: All granularities (2–1000) per region, 3 subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    class_labels = [2, 4, 8, 16, 32, 64, 1000]
    n_bars = len(class_labels)
    bar_x = np.arange(n_bars)
    bar_width = 0.65

    for ax_idx, (ax, region) in enumerate(zip(axes, regions)):
        means, sems, all_seeds = [], [], []
        for n_cls in class_labels:
            row = summary[(summary['region'] == region) & (summary['n_classes'] == n_cls)]
            if len(row) > 0:
                means.append(row['mean'].values[0])
                sems.append(row['sem'].values[0])
                all_seeds.append(row['seeds'].values[0])
            else:
                means.append(0); sems.append(0); all_seeds.append([])
        means = np.array(means)
        sems = np.array(sems)

        blues = plt.cm.Blues(np.linspace(0.3, 0.75, 6))
        colors = list(blues) + [matplotlib.colors.to_rgba(BASELINE_COLOR)]
        ax.bar(bar_x, means, bar_width, color=colors, edgecolor='white', linewidth=0.3,
               yerr=sems, capsize=1.8, error_kw={'linewidth': 0.5, 'capthick': 0.5})

        np.random.seed(42)
        for i in range(n_bars):
            if all_seeds[i]:
                jitter = np.random.uniform(-0.1, 0.1, size=len(all_seeds[i]))
                ax.scatter(bar_x[i] + jitter, all_seeds[i], s=6, color=POINT_COLOR,
                           zorder=5, alpha=0.6, linewidths=0.2, edgecolors='white')

        ax.axhline(y=means[-1], color=BASELINE_COLOR, linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
        ax.set_xticks(bar_x)
        ax.set_xticklabels([str(c) for c in class_labels], rotation=0)
        ax.set_xlabel('Number of classes', labelpad=3)
        ax.set_title(region, fontweight='bold', pad=6)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_ylim(0, max(means) + max(sems) + 0.03)

    axes[0].set_ylabel('RSA Score', labelpad=4)
    plt.tight_layout(w_pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot 2 saved: {out_path}")


if __name__ == '__main__':
    summary = load_and_summarize(CSV_PATH, CHECKPOINT_DIRS)
    plot_best_coarse_vs_1k(summary, REGIONS, 'figures/tvsd_best_coarse_vs_1k.png')
    plot_all_granularities(summary, REGIONS, 'figures/tvsd_all_granularities_by_region.png')
