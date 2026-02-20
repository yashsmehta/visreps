"""Box plot comparing brain alignment across PCA label architectures.

Two modes:
  METRIC = "rsa"      — RSA scores, 4 architectures, with paired t-test brackets
  METRIC = "encoding" — Encoding scores, 2 architectures, no significance

Usage:
    source .venv/bin/activate && python plotters/box_plot_architectures.py [--region REGION]
"""

import sys
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# CONFIGURATION — change METRIC to switch between RSA and encoding
# ============================================================================
METRIC = "rsa"  # "rsa" or "encoding"

region_to_plot = 'ventral visual stream' if METRIC == "rsa" else 'early visual stream'

METRIC_CONFIG = {
    "rsa": {
        "layer_by_region": {
            'early visual stream': 'conv5',
            'ventral visual stream': 'fc1',
        },
        "arch_folder_map": {
            'AlexNet': 'pca_labels_alexnet',
            'ViT': 'pca_labels_vit',
            'DINO': 'pca_labels_dino',
            'CLIP': 'pca_labels_clip',
        },
        "arch_order": ['AlexNet', 'ViT', 'DINO', 'CLIP'],
        "coarsegrain_csv": "all_rsa_coarsegrain.csv",
        "k1k_csv": "all_rsa_1k.csv",
        "ylabel": "RSA Score",
        "enable_significance": True,
        "figsize": (7, 5),
    },
    "encoding": {
        "layer_by_region": {
            'early visual stream': 'conv4',
            'ventral visual stream': 'fc2',
        },
        "arch_folder_map": {
            'AlexNet': 'pca_labels_alexnet',
            'ViT': 'pca_labels_vit',
        },
        "arch_order": ['AlexNet', 'ViT'],
        "coarsegrain_csv": "all_encoding_coarsegrain.csv",
        "k1k_csv": "all_encoding_1k.csv",
        "ylabel": "Encoding Score",
        "enable_significance": False,
        "figsize": (6, 5),
    },
}


def find_best_pca_class(df, arch_folder, region, layer):
    """Find the PCA class count with the highest mean score for a given architecture."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]
    if len(df_filtered) == 0:
        return None, None
    mean_by_class = df_filtered.groupby('pca_n_classes')['score'].mean()
    best_n_classes = mean_by_class.idxmax()
    return best_n_classes, mean_by_class.max()


def get_scores_for_boxplot(df, arch_folder, region, layer, n_classes):
    """Get scores averaged across seeds for each subject."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes)
    ]
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means.values


def get_scores_for_ttest(df, arch_folder, region, layer, n_classes):
    """Get all scores (seed x subject combinations) for paired t-test."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes)
    ]
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


def get_1k_scores_for_boxplot(df, region, layer):
    """Get 1K scores averaged across seeds for each subject."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means.values


def get_1k_scores_for_ttest(df, region, layer):
    """Get all 1K scores (seed x subject combinations) for paired t-test."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


def plot_boxplot(data_dict, labels, region, layer, out_png, significance_results=None):
    """Create Nature-style boxplot with optional significance markers."""
    cfg = METRIC_CONFIG[METRIC]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    box_data = [data_dict[label]['boxplot'] for label in labels]

    # Color palette: grey for 1K, then muted colors for architectures
    colors = ['#7f7f7f']
    arch_colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f']
    colors.extend(arch_colors[:len(labels)-1])

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=4, alpha=0.6))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')

    # Individual data points
    for i, label in enumerate(labels):
        y = data_dict[label]['boxplot']
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.8, s=35, c='white', edgecolors='black',
                   linewidths=0.8, zorder=3)

    # Y-axis limits
    y_min = min([min(data_dict[label]['boxplot']) for label in labels])
    y_max = max([max(data_dict[label]['boxplot']) for label in labels])
    y_range = y_max - y_min

    if significance_results:
        n_pca_conditions = len([l for l in labels if l != '1K'])
        y_padding = y_range * (0.08 * n_pca_conditions + 0.1)
    else:
        y_padding = y_range * 0.1

    y_bottom = np.floor(y_min * 20) / 20
    y_top = y_max + y_padding
    y_top = np.ceil(y_top * 20) / 20
    ax.set_ylim(y_bottom, y_top)

    y_ticks = np.arange(y_bottom, y_top + 0.01, 0.05)
    ax.set_yticks(y_ticks)

    # Significance brackets (RSA mode only)
    if significance_results:
        bracket_base = y_max + y_range * 0.05
        bracket_step = y_range * 0.08
        pca_idx = 0
        for i, label in enumerate(labels):
            if label != '1K' and label in significance_results:
                p_val = significance_results[label]['p_value']
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'n.s.'

                x1, x2 = 1, i + 1
                y_bracket = bracket_base + pca_idx * bracket_step
                tick_height = y_range * 0.015
                ax.plot([x1, x1], [y_bracket - tick_height, y_bracket], color='black', linewidth=1.0)
                ax.plot([x1, x2], [y_bracket, y_bracket], color='black', linewidth=1.0)
                ax.plot([x2, x2], [y_bracket - tick_height, y_bracket], color='black', linewidth=1.0)
                ax.text((x1 + x2) / 2, y_bracket + y_range * 0.015, sig_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                pca_idx += 1

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha='center')

    ax.set_ylabel(cfg["ylabel"], fontsize=11, fontweight='medium')
    ax.set_xlabel('PCA Label Source', fontsize=11, fontweight='medium')

    region_name = region.replace('visual stream', 'Visual Stream').title()
    ax.set_title(f'{region_name}\n(Layer: {layer.upper()})',
                 fontsize=12, fontweight='bold', pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', length=4, width=1)
    ax.tick_params(axis='x', which='major', bottom=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved plot to {out_png}")


# --- Main Script ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Box plot comparing brain alignment across PCA architectures')
    parser.add_argument('--region', type=str, default=region_to_plot,
                        choices=['early visual stream', 'ventral visual stream'],
                        help='Brain region to analyze')
    args = parser.parse_args()
    region_to_plot = args.region

    cfg = METRIC_CONFIG[METRIC]
    base_log_path = 'logs/'
    epoch_to_plot = 20

    layer = cfg["layer_by_region"][region_to_plot]
    region_slug = region_to_plot.lower().replace(' ', '_')
    prefix = "architecture_comparison" if METRIC == "rsa" else "encoding_architecture_comparison"
    out_png = f"plotters/figures/{prefix}_{region_slug}.png"

    print(f"Loading data for region: {region_to_plot}, layer: {layer}")

    df_coarse = pd.read_csv(os.path.join(base_log_path, cfg["coarsegrain_csv"]))
    df_coarse['layer'] = df_coarse['layer'].str.lower()
    df_coarse = df_coarse[df_coarse['epoch'] == epoch_to_plot]

    df_1k = pd.read_csv(os.path.join(base_log_path, cfg["k1k_csv"]))
    df_1k['layer'] = df_1k['layer'].str.lower()
    df_1k = df_1k[df_1k['epoch'] == epoch_to_plot]

    # Find best PCA class for each architecture
    best_pca_classes = {}
    for arch_name, arch_folder in cfg["arch_folder_map"].items():
        best_n, best_score = find_best_pca_class(df_coarse, arch_folder, region_to_plot, layer)
        if best_n is not None:
            best_pca_classes[arch_name] = best_n
            print(f"{arch_name}: best PCA classes = {int(best_n)} (mean score = {best_score:.4f})")
        else:
            print(f"{arch_name}: no data found")

    # Collect scores
    data_dict = {}
    labels = []

    # 1K baseline
    scores_1k_box = get_1k_scores_for_boxplot(df_1k, region_to_plot, layer)
    if len(scores_1k_box) > 0:
        data_dict['1K'] = {'boxplot': scores_1k_box, 'n_classes': 1000}
        labels.append('1K')
        print(f"\n--- 1K Baseline ---")
        print(f"  Boxplot: {len(scores_1k_box)} subjects")
        print(f"  Subject means: {[f'{x:.4f}' for x in scores_1k_box]}")

        if cfg["enable_significance"]:
            data_dict['1K']['ttest'] = get_1k_scores_for_ttest(df_1k, region_to_plot, layer)
            print(f"  T-test: {len(data_dict['1K']['ttest'])} samples")

    # PCA architectures
    for arch_name in cfg["arch_order"]:
        if arch_name not in best_pca_classes:
            continue

        arch_folder = cfg["arch_folder_map"][arch_name]
        n_classes = best_pca_classes[arch_name]

        scores_box = get_scores_for_boxplot(df_coarse, arch_folder, region_to_plot, layer, n_classes)
        if len(scores_box) > 0:
            label = f"{arch_name} ({int(n_classes)})"
            data_dict[label] = {'boxplot': scores_box, 'n_classes': n_classes}
            labels.append(label)
            print(f"\n--- {label} ---")
            print(f"  Boxplot: {len(scores_box)} subjects")
            print(f"  Subject means: {[f'{x:.4f}' for x in scores_box]}")

            if cfg["enable_significance"]:
                data_dict[label]['ttest'] = get_scores_for_ttest(df_coarse, arch_folder, region_to_plot, layer, n_classes)

    # Paired t-tests (RSA mode only)
    significance_results = {}
    if cfg["enable_significance"] and '1K' in data_dict and 'ttest' in data_dict['1K']:
        print("\n=== Paired t-tests vs ImageNet-1K ===")
        scores_1k = data_dict['1K']['ttest']
        print(f"1K baseline: mean={np.mean(scores_1k):.4f}, std={np.std(scores_1k):.4f}, n={len(scores_1k)}")
        for label in labels:
            if label == '1K' or 'ttest' not in data_dict[label]:
                continue
            scores = data_dict[label]['ttest']
            if len(scores) == len(scores_1k):
                t_stat, p_val = stats.ttest_rel(scores, scores_1k)
                sig_mark = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                significance_results[label] = {'t_stat': t_stat, 'p_value': p_val}
                diff = np.mean(scores) - np.mean(scores_1k)
                print(f"{label:20s}: mean={np.mean(scores):.4f}, diff={diff:+.4f}, t={t_stat:7.3f}, p={p_val:.6f} {sig_mark}")

    # Plot
    plot_boxplot(data_dict, labels, region_to_plot, layer, out_png,
                 significance_results if cfg["enable_significance"] else None)

    print("\n=== Summary ===")
    print(f"Region: {region_to_plot}")
    print(f"Layer: {layer}")
    print(f"Best PCA classes per architecture:")
    for arch, n in best_pca_classes.items():
        print(f"  {arch}: {int(n)}")
