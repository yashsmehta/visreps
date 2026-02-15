"""
Paired box plot comparing Global vs Hierarchical ViT-based PCA labels.

Creates a figure with 2 subplots:
1. Early Visual Stream (conv4 layer)
2. Ventral Visual Stream (fc2 layer)

Each subplot shows box plots for both methods with lines connecting
the same subject across methods (paired comparison).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = "logs/vit_global_v_hierarchical.csv"
OUTPUT_PATH = "plotters/post-neurips/vit_global_vs_hierarchical_paired.png"

# Subplot configurations: (region, layer, title)
SUBPLOT_CONFIGS = [
    ("early visual stream", "conv4", "Early Visual Stream (Conv4)"),
    ("ventral visual stream", "fc2", "Ventral Visual Stream (FC2)"),
]

# Method identification
COMPARE_COLUMN = "checkpoint_dir"
METHOD_A_VALUE = "model_checkpoints/alexnet_hierarchical_vit"
METHOD_B_VALUE = "model_checkpoints/alexnet_global_vit"
METHOD_A_LABEL = "Hierarchical"
METHOD_B_LABEL = "Global"

# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Load data
    df = pd.read_csv(CSV_PATH)
    df['layer'] = df['layer'].str.lower()
    df['region'] = df['region'].str.lower()

    # Set up the figure
    sns.set_theme(style='ticks', context='paper', font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Colors for the two methods
    colors = {
        METHOD_A_LABEL: '#1f77b4',  # Blue
        METHOD_B_LABEL: '#ff7f0e',  # Orange
    }

    for ax_idx, (region, layer, title) in enumerate(SUBPLOT_CONFIGS):
        ax = axes[ax_idx]

        # Filter data for this subplot
        df_region = df[
            (df['region'] == region) &
            (df['layer'] == layer)
        ]

        # Separate by method
        df_a = df_region[df_region[COMPARE_COLUMN] == METHOD_A_VALUE].copy()
        df_b = df_region[df_region[COMPARE_COLUMN] == METHOD_B_VALUE].copy()

        # Ensure we have data for both methods
        if len(df_a) == 0 or len(df_b) == 0:
            print(f"Warning: Missing data for {region}, {layer}")
            continue

        # Sort by subject_idx to ensure proper pairing
        df_a = df_a.sort_values('subject_idx').reset_index(drop=True)
        df_b = df_b.sort_values('subject_idx').reset_index(drop=True)

        # Get scores
        scores_a = df_a['score'].values
        scores_b = df_b['score'].values
        subject_ids = df_a['subject_idx'].values

        print(f"\n{title}:")
        print(f"  {METHOD_A_LABEL}: mean={np.mean(scores_a):.4f}, std={np.std(scores_a):.4f}")
        print(f"  {METHOD_B_LABEL}: mean={np.mean(scores_b):.4f}, std={np.std(scores_b):.4f}")

        # Create DataFrame for seaborn boxplot
        plot_data = pd.DataFrame({
            'Method': [METHOD_A_LABEL] * len(scores_a) + [METHOD_B_LABEL] * len(scores_b),
            'Score': np.concatenate([scores_a, scores_b]),
            'Subject': np.concatenate([subject_ids, subject_ids])
        })

        # Draw box plots
        box_positions = {METHOD_A_LABEL: 0, METHOD_B_LABEL: 1}

        for method, pos in box_positions.items():
            method_scores = plot_data[plot_data['Method'] == method]['Score'].values
            bp = ax.boxplot(
                [method_scores],
                positions=[pos],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
            )
            bp['boxes'][0].set_facecolor(colors[method])
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_linewidth(1.5)
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(2)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)

        # Draw individual points and connecting lines
        jitter = 0.08
        np.random.seed(42)  # For reproducible jitter

        for i, subj in enumerate(subject_ids):
            # Add small jitter to x positions for visibility
            x_a = 0 + np.random.uniform(-jitter, jitter)
            x_b = 1 + np.random.uniform(-jitter, jitter)

            # Draw connecting line
            ax.plot(
                [x_a, x_b],
                [scores_a[i], scores_b[i]],
                color='gray',
                alpha=0.5,
                linewidth=1.2,
                zorder=1
            )

            # Draw points
            ax.scatter(x_a, scores_a[i], color=colors[METHOD_A_LABEL],
                      s=60, edgecolor='black', linewidth=0.8, zorder=2)
            ax.scatter(x_b, scores_b[i], color=colors[METHOD_B_LABEL],
                      s=60, edgecolor='black', linewidth=0.8, zorder=2)

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels([METHOD_A_LABEL, METHOD_B_LABEL], fontsize=12, fontweight='bold')
        ax.set_ylabel('Brain Similarity (RSA)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Paired t-test for significance
        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")

        # Set y-axis limits with some padding (extra padding at top for significance marker)
        all_scores = np.concatenate([scores_a, scores_b])
        y_max = max(all_scores) + 0.05  # Extra space for significance annotation
        ax.set_ylim(0, y_max)

        # Add significance annotation if p < 0.05
        if p_val < 0.05:
            # Draw bracket and star
            bracket_y = max(all_scores) + 0.015
            star_y = bracket_y + 0.012

            # Horizontal bracket line
            ax.plot([0, 0, 1, 1], [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005],
                    color='black', linewidth=1.5)

            # Significance star(s)
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            else:
                sig_text = '*'

            ax.text(0.5, star_y, sig_text, ha='center', va='bottom',
                    fontsize=16, fontweight='bold', color='black')

    # Adjust layout
    plt.tight_layout(pad=2.0)

    # Save figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nPlot saved to: {OUTPUT_PATH}")
