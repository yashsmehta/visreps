"""
RSA score vs number of coarse-grained classes.
Single plot for a specified region and architecture.
Shows individual subjects connected across class counts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================================
# CONFIGURATION - Change these to switch region/architecture
# ============================================================================
REGION = 'ventral visual stream'  # Options: 'early visual stream', 'ventral visual stream'
ARCHITECTURE = 'ViT'  # Options: 'AlexNet', 'ViT', 'DINO', 'CLIP'

# Layer mapping by region
LAYER_BY_REGION = {
    'early visual stream': 'conv4',
    'ventral visual stream': 'fc2',
}

# Architecture folder mapping
ARCH_FOLDER_MAP = {
    'AlexNet': 'pca_labels_alexnet',
    'ViT': 'pca_labels_vit',
    'DINO': 'pca_labels_dino',
    'CLIP': 'pca_labels_clip',
}

# Class counts to plot (in order)
CLASS_COUNTS = [2, 4, 8, 16, 32, 64]
EPOCH = 20

# ============================================================================
# Nature-style plot settings
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================================
# Paths
# ============================================================================
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# Data loading and processing functions
# ============================================================================

def get_subject_means_coarse(df, arch_folder, region, layer, n_classes):
    """Get scores averaged across seeds for each subject."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes) &
        (df['epoch'] == EPOCH)
    ]

    # Average across seeds for each subject
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means


def get_subject_means_1k(df, region, layer):
    """Get 1K scores averaged across seeds for each subject."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['epoch'] == EPOCH)
    ]

    # Average across seeds for each subject
    subject_means = df_filtered.groupby('subject_idx')['score'].mean()
    return subject_means


def get_all_scores_coarse(df, arch_folder, region, layer, n_classes):
    """Get all scores (seed × subject) for paired t-test."""
    df_filtered = df[
        (df['pca_labels_folder'] == arch_folder) &
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['pca_n_classes'] == n_classes) &
        (df['epoch'] == EPOCH)
    ]
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


def get_all_scores_1k(df, region, layer):
    """Get all 1K scores (seed × subject) for paired t-test."""
    df_filtered = df[
        (df['region'].str.lower() == region.lower()) &
        (df['layer'].str.lower() == layer.lower()) &
        (df['epoch'] == EPOCH)
    ]
    df_sorted = df_filtered.sort_values(['seed', 'subject_idx'])
    return df_sorted['score'].values


# ============================================================================
# Main script
# ============================================================================

if __name__ == "__main__":
    # Get layer and folder for current config
    layer = LAYER_BY_REGION[REGION]
    arch_folder = ARCH_FOLDER_MAP[ARCHITECTURE]

    print(f"Region: {REGION}")
    print(f"Architecture: {ARCHITECTURE} ({arch_folder})")
    print(f"Layer: {layer}")

    # Load data
    df_coarse = pd.read_csv(LOGS_DIR / "all_rsa_coarsegrain.csv")
    df_1k = pd.read_csv(LOGS_DIR / "all_rsa_1k.csv")

    # Collect data for each condition
    # Structure: {condition_label: {'subject_means': Series, 'all_scores': array}}
    data = {}
    x_labels = []

    # Coarse-grained conditions
    for n_classes in CLASS_COUNTS:
        label = str(n_classes)
        subject_means = get_subject_means_coarse(df_coarse, arch_folder, REGION, layer, n_classes)
        all_scores = get_all_scores_coarse(df_coarse, arch_folder, REGION, layer, n_classes)

        if len(subject_means) > 0:
            data[label] = {
                'subject_means': subject_means,
                'all_scores': all_scores
            }
            x_labels.append(label)
            print(f"  {n_classes} classes: {len(subject_means)} subjects, mean={subject_means.mean():.4f}")

    # 1K baseline
    subject_means_1k = get_subject_means_1k(df_1k, REGION, layer)
    all_scores_1k = get_all_scores_1k(df_1k, REGION, layer)

    if len(subject_means_1k) > 0:
        data['1K'] = {
            'subject_means': subject_means_1k,
            'all_scores': all_scores_1k
        }
        x_labels.append('1K')
        print(f"  1K: {len(subject_means_1k)} subjects, mean={subject_means_1k.mean():.4f}")

    # Get common subjects across all conditions
    common_subjects = None
    for label in x_labels:
        subjects = set(data[label]['subject_means'].index)
        if common_subjects is None:
            common_subjects = subjects
        else:
            common_subjects = common_subjects.intersection(subjects)

    common_subjects = sorted(list(common_subjects))
    print(f"\nCommon subjects: {common_subjects}")

    # ========================================================================
    # Statistical tests (paired t-test vs 1K)
    # ========================================================================
    print("\n=== Paired t-tests vs 1K ===")
    significance = {}

    for label in x_labels:
        if label == '1K':
            continue

        scores = data[label]['all_scores']
        scores_1k = data['1K']['all_scores']

        if len(scores) == len(scores_1k):
            t_stat, p_val = stats.ttest_rel(scores, scores_1k)
            mean_diff = np.mean(scores) - np.mean(scores_1k)

            # Only mark significant if p < 0.05 AND mean is greater than 1K
            if p_val < 0.05 and mean_diff > 0:
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                else:
                    sig_text = '*'
                significance[label] = {'p_value': p_val, 'sig_text': sig_text, 'mean_diff': mean_diff}
                print(f"  {label}: diff={mean_diff:+.4f}, p={p_val:.6f} {sig_text}")
            else:
                print(f"  {label}: diff={mean_diff:+.4f}, p={p_val:.6f} (not shown: {'below 1K' if mean_diff <= 0 else 'n.s.'})")

    # ========================================================================
    # Create plot
    # ========================================================================
    fig, ax = plt.subplots(figsize=(5, 4))

    # Create x positions with a gap before 1K
    n_coarse = len([l for l in x_labels if l != '1K'])
    x_positions = []
    for i, label in enumerate(x_labels):
        if label == '1K':
            x_positions.append(n_coarse + 0.7)  # Add gap before 1K
        else:
            x_positions.append(len(x_positions))
    x_positions = np.array(x_positions)

    # Colors: 6 shades of blue for coarse-grained, gray for 1K
    blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    gray = "#7f7f7f"

    colors = []
    for label in x_labels:
        if label == '1K':
            colors.append(gray)
        else:
            idx = CLASS_COUNTS.index(int(label))
            colors.append(blue_shades[idx])

    # Prepare box plot data (subject means)
    box_data = []
    for label in x_labels:
        means = data[label]['subject_means']
        # Reindex to common subjects to ensure alignment
        box_data.append(means.loc[common_subjects].values)

    # Draw box plots
    bp = ax.boxplot(box_data, positions=x_positions, patch_artist=True, widths=0.5,
                    boxprops=dict(linewidth=1.0),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0),
                    medianprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Style boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    # Draw connecting lines for each subject (subtle)
    for subj in common_subjects:
        y_values = []
        for label in x_labels:
            y_values.append(data[label]['subject_means'].loc[subj])

        ax.plot(x_positions, y_values, color='gray', alpha=0.25, linewidth=0.8, zorder=1)

    # Draw individual subject points
    np.random.seed(42)
    for i, label in enumerate(x_labels):
        means = data[label]['subject_means']
        y = means.loc[common_subjects].values
        x_jitter = np.random.normal(x_positions[i], 0.06, size=len(y))
        ax.scatter(x_jitter, y, s=25, c='white', edgecolors='black',
                   linewidths=0.7, zorder=3, alpha=0.9)

    # Add significance markers
    y_max = max([data[label]['subject_means'].loc[common_subjects].max() for label in x_labels])
    y_range = y_max - min([data[label]['subject_means'].loc[common_subjects].min() for label in x_labels])

    # Position of 1K on x-axis
    x_1k = x_labels.index('1K')

    bracket_base = y_max + y_range * 0.08
    bracket_step = y_range * 0.12

    sig_idx = 0
    for label in x_labels:
        if label in significance:
            x_cond = x_labels.index(label)
            y_bracket = bracket_base + sig_idx * bracket_step

            # Draw bracket
            tick_height = y_range * 0.02
            ax.plot([x_cond, x_cond], [y_bracket - tick_height, y_bracket], color='black', linewidth=0.8)
            ax.plot([x_cond, x_1k], [y_bracket, y_bracket], color='black', linewidth=0.8)
            ax.plot([x_1k, x_1k], [y_bracket - tick_height, y_bracket], color='black', linewidth=0.8)

            # Significance text
            ax.text((x_cond + x_1k) / 2, y_bracket + y_range * 0.02,
                    significance[label]['sig_text'],
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

            sig_idx += 1

    # Labels and formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight='bold')
    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("RSA Score")

    # Title
    region_title = REGION.replace('visual stream', 'Visual Stream').title()
    ax.set_title(f"{region_title} - {ARCHITECTURE}\n(Layer: {layer})", fontweight='bold')

    # Y-axis limits with padding for significance brackets
    y_min = min([data[label]['subject_means'].loc[common_subjects].min() for label in x_labels])
    y_padding = y_range * (0.15 + 0.12 * len(significance))
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_padding)

    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Set x-axis limits to frame the gap nicely
    ax.set_xlim(-0.5, x_positions[-1] + 0.5)

    plt.tight_layout()

    # Save
    region_short = 'EVC' if 'early' in REGION.lower() else 'VVS'
    arch_short = ARCHITECTURE.lower()
    output_path = OUTPUT_DIR / f"rsa_by_classes_{region_short}_{arch_short}.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved to {output_path}")
    plt.show()
