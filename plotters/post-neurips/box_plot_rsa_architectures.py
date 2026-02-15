import sys
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducible scatter jitter

# ============================================================================
# CONFIGURATION
# ============================================================================
region_to_plot = 'ventral visual stream'  # Options: 'early visual stream', 'ventral visual stream'

# Layer to use based on region
LAYER_BY_REGION = {
    'early visual stream': 'conv5',
    'ventral visual stream': 'fc1'
}

# Architecture folder mapping
ARCH_FOLDER_MAP = {
    'AlexNet': 'pca_labels_alexnet',
    'ViT': 'pca_labels_vit',
    'DINO': 'pca_labels_dino',
    'CLIP': 'pca_labels_clip',
}

# Order for plotting (1K first, then architectures)
PLOT_ORDER = ['1K', 'AlexNet', 'ViT', 'DINO', 'CLIP']


def find_best_pca_class(df, arch_folder, region, layer):
    """Find the PCA class count with the highest mean score for a given architecture."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]

    if len(df_filtered) == 0:
        return None, None

    # Group by pca_n_classes and compute mean score across all seeds and subjects
    mean_by_class = df_filtered.groupby('pca_n_classes')['score'].mean()
    best_n_classes = mean_by_class.idxmax()

    return best_n_classes, mean_by_class.max()


def get_scores_for_boxplot(df, arch_folder, region, layer, n_classes):
    """Get scores averaged across seeds for each subject (for boxplot display)."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes)
    ]

    # Average across seeds for each subject
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means.values


def get_scores_for_ttest(df, arch_folder, region, layer, n_classes):
    """Get all scores (seed × subject combinations) for paired t-test."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes)
    ]

    # Sort by seed and subject_idx to ensure pairing
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


def get_1k_scores_for_boxplot(df, region, layer):
    """Get 1K scores averaged across seeds for each subject."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]

    # Average across seeds for each subject
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means.values


def get_1k_scores_for_ttest(df, region, layer):
    """Get all 1K scores (seed × subject combinations) for paired t-test."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower())
    ]

    # Sort by seed and subject_idx to ensure pairing
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


def plot_boxplot(data_dict, labels, region, layer, out_png, significance_results):
    """Create Nature-style boxplot with significance markers."""

    # Set up Nature-style formatting
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

    fig, ax = plt.subplots(figsize=(7, 5))

    # Prepare data for boxplot
    box_data = [data_dict[label]['boxplot'] for label in labels]

    # Nature-style color palette (muted, professional)
    colors = ['#7f7f7f']  # Grey for 1K baseline
    arch_colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f']  # Muted blue, orange, green, red
    colors.extend(arch_colors[:len(labels)-1])

    # Create boxplot with Nature-style formatting
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=4, alpha=0.6))

    # Style the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')

    # Add individual data points (strip plot style)
    for i, label in enumerate(labels):
        y = data_dict[label]['boxplot']
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.8, s=35, c='white', edgecolors='black',
                   linewidths=0.8, zorder=3)

    # Calculate y-axis limits with space for significance markers
    y_min = min([min(data_dict[label]['boxplot']) for label in labels])
    y_max = max([max(data_dict[label]['boxplot']) for label in labels])
    y_range = y_max - y_min

    # Calculate number of PCA conditions (excluding 1K) for bracket spacing
    n_pca_conditions = len([l for l in labels if l != '1K'])
    y_padding = y_range * (0.08 * n_pca_conditions + 0.1)  # Dynamic padding based on conditions

    # Set y-axis limits with 0.05 tick increments
    y_bottom = np.floor(y_min * 20) / 20  # Round down to nearest 0.05
    y_top = y_max + y_padding
    y_top = np.ceil(y_top * 20) / 20  # Round up to nearest 0.05
    ax.set_ylim(y_bottom, y_top)

    # Set y-axis ticks at 0.05 increments
    y_ticks = np.arange(y_bottom, y_top + 0.01, 0.05)
    ax.set_yticks(y_ticks)

    # Add significance brackets comparing each condition to 1K
    bracket_base = y_max + y_range * 0.05
    bracket_step = y_range * 0.08

    pca_idx = 0  # Track PCA condition index for bracket stacking
    for i, label in enumerate(labels):
        if label != '1K' and label in significance_results:
            p_val = significance_results[label]['p_value']

            # Determine significance level
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'n.s.'

            # Draw bracket from 1K (position 1) to current position
            x1, x2 = 1, i + 1
            y_bracket = bracket_base + pca_idx * bracket_step

            # Draw bracket: vertical ticks at ends connected by horizontal line
            tick_height = y_range * 0.015
            ax.plot([x1, x1], [y_bracket - tick_height, y_bracket], color='black', linewidth=1.0)
            ax.plot([x1, x2], [y_bracket, y_bracket], color='black', linewidth=1.0)
            ax.plot([x2, x2], [y_bracket - tick_height, y_bracket], color='black', linewidth=1.0)

            # Significance text above bracket
            ax.text((x1 + x2) / 2, y_bracket + y_range * 0.015, sig_text,
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

            pca_idx += 1

    # Set x-axis labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, ha='center')

    # Labels and title
    ax.set_ylabel('RSA Score', fontsize=11, fontweight='medium')
    ax.set_xlabel('PCA Label Source', fontsize=11, fontweight='medium')

    # Title with region and layer info
    region_name = region.replace('visual stream', 'Visual Stream').title()
    ax.set_title(f'{region_name}\n(Layer: {layer.upper()})',
                 fontsize=12, fontweight='bold', pad=10)

    # Remove top and right spines (Nature style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Subtle horizontal grid lines only
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust tick parameters
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

    # Override config with command-line arg if provided
    region_to_plot = args.region

    # ---------------- config ----------------
    base_log_path = 'logs/'
    coarsegrain_csv = 'all_rsa_coarsegrain.csv'
    k1k_csv = 'all_rsa_1k.csv'
    epoch_to_plot = 20

    layer = LAYER_BY_REGION[region_to_plot]
    region_slug = region_to_plot.lower().replace(' ', '_')
    out_png = f"plotters/post-neurips/architecture_comparison_{region_slug}.png"

    # ---------------- load data ----------------
    print(f"Loading data for region: {region_to_plot}, layer: {layer}")

    df_coarse = pd.read_csv(os.path.join(base_log_path, coarsegrain_csv))
    df_coarse['layer'] = df_coarse['layer'].str.lower()
    df_coarse = df_coarse[df_coarse['epoch'] == epoch_to_plot]

    df_1k = pd.read_csv(os.path.join(base_log_path, k1k_csv))
    df_1k['layer'] = df_1k['layer'].str.lower()
    df_1k = df_1k[df_1k['epoch'] == epoch_to_plot]

    # ---------------- find best PCA class for each architecture ----------------
    best_pca_classes = {}
    for arch_name, arch_folder in ARCH_FOLDER_MAP.items():
        best_n, best_score = find_best_pca_class(df_coarse, arch_folder, region_to_plot, layer)
        if best_n is not None:
            best_pca_classes[arch_name] = best_n
            print(f"{arch_name}: best PCA classes = {int(best_n)} (mean score = {best_score:.4f})")
        else:
            print(f"{arch_name}: no data found")

    # ---------------- collect scores ----------------
    data_dict = {}
    labels = []

    # 1K baseline
    scores_1k_box = get_1k_scores_for_boxplot(df_1k, region_to_plot, layer)
    scores_1k_ttest = get_1k_scores_for_ttest(df_1k, region_to_plot, layer)

    if len(scores_1k_box) > 0:
        data_dict['1K'] = {
            'boxplot': scores_1k_box,
            'ttest': scores_1k_ttest,
            'n_classes': 1000
        }
        labels.append('1K')
        print(f"\n--- 1K Baseline ---")
        print(f"  Boxplot: {len(scores_1k_box)} subjects (mean across seeds per subject)")
        print(f"  T-test:  {len(scores_1k_ttest)} samples (3 seeds × 8 subjects = 24 expected)")
        print(f"  Subject means: {[f'{x:.4f}' for x in scores_1k_box]}")

    # PCA architectures (order: AlexNet, ViT, CLIP, DINO)
    for arch_name in ['AlexNet', 'ViT', 'CLIP', 'DINO']:
        if arch_name not in best_pca_classes:
            continue

        arch_folder = ARCH_FOLDER_MAP[arch_name]
        n_classes = best_pca_classes[arch_name]

        scores_box = get_scores_for_boxplot(df_coarse, arch_folder, region_to_plot, layer, n_classes)
        scores_ttest = get_scores_for_ttest(df_coarse, arch_folder, region_to_plot, layer, n_classes)

        if len(scores_box) > 0:
            label = f"{arch_name} ({int(n_classes)})"
            data_dict[label] = {
                'boxplot': scores_box,
                'ttest': scores_ttest,
                'n_classes': n_classes
            }
            labels.append(label)
            print(f"\n--- {label} ---")
            print(f"  Boxplot: {len(scores_box)} subjects (mean across seeds per subject)")
            print(f"  T-test:  {len(scores_ttest)} samples (3 seeds × 8 subjects = 24 expected)")
            print(f"  Subject means: {[f'{x:.4f}' for x in scores_box]}")

    # ---------------- verify t-test pairing ----------------
    print("\n=== Verifying T-test Sample Pairing ===")
    # Check that 1K data has correct (seed, subject) combinations
    df_1k_check = df_1k[
        (df_1k['region'].str.lower() == region_to_plot.lower()) &
        (df_1k['layer'].str.lower() == layer.lower())
    ].sort_values(['seed', 'subject_idx'])
    print(f"1K: seeds={sorted(df_1k_check['seed'].unique())}, subjects={sorted(df_1k_check['subject_idx'].unique())}")
    print(f"1K: total rows = {len(df_1k_check)} (expected: 3 seeds × 8 subjects = 24)")

    # ---------------- paired t-tests vs 1K ----------------
    print("\n=== Paired t-tests vs ImageNet-1K (24 paired samples: 3 seeds x 8 subjects) ===")
    significance_results = {}

    scores_1k = data_dict['1K']['ttest']
    print(f"1K baseline: mean={np.mean(scores_1k):.4f}, std={np.std(scores_1k):.4f}, n={len(scores_1k)}")

    for label in labels:
        if label == '1K':
            continue

        scores = data_dict[label]['ttest']

        if len(scores) == len(scores_1k):
            t_stat, p_val = stats.ttest_rel(scores, scores_1k)
            if p_val < 0.001:
                sig_mark = "***"
            elif p_val < 0.01:
                sig_mark = "**"
            elif p_val < 0.05:
                sig_mark = "*"
            else:
                sig_mark = "n.s."
            significance_results[label] = {'t_stat': t_stat, 'p_value': p_val}
            diff = np.mean(scores) - np.mean(scores_1k)
            print(f"{label:20s}: mean={np.mean(scores):.4f}, diff={diff:+.4f}, t={t_stat:7.3f}, p={p_val:.6f} {sig_mark}")
        else:
            print(f"{label:20s}: ERROR - sample size mismatch ({len(scores)} vs {len(scores_1k)})")

    # ---------------- plot ----------------
    plot_boxplot(data_dict, labels, region_to_plot, layer, out_png, significance_results)

    print("\n=== Summary ===")
    print(f"Region: {region_to_plot}")
    print(f"Layer: {layer}")
    print(f"Best PCA classes per architecture:")
    for arch, n in best_pca_classes.items():
        print(f"  {arch}: {int(n)}")
