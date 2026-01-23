import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# Import from the utils module
from plotters.utils import plotter_utils as plt_utils

# Set seaborn style for better aesthetics
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

def create_individual_subject_plots(data_df, regions, layer_order, dataset_name, metric_name):
    # Define color scheme
    colors = {
        'initial': '#7f8c8d',    # Elegant gray
        'final': '#c0392b',      # Dark red/orange for full model
    }

    # Create continuous color scale for PCA variants using blues
    n_pca_variants = 6
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants))
    pca_colors = dict(zip([2, 4, 8, 16, 32, 64], blues))

    # Get unique subjects from the provided data_df
    subjects = sorted(data_df['subject_idx'].unique())
    n_subjects = len(subjects)

    # Grid dimensions for 8 subjects
    n_rows = 2
    n_cols = 4

    for region in regions:
        print(f"\nProcessing region: {region}")

        # Create figure with extra height for legend at bottom
        fig = plt.figure(figsize=(20, 13))  # Increased height for legend space

        # Create main grid for plots with more space at bottom
        gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=[1, 1], bottom=0.15)  # Add space at bottom

        # Calculate Y-axis limits for the current region across all relevant conditions and subjects
        # Assumes data_df contains 'dataset', 'compare_rsm_correlation', 'region', 'epoch', 'pca_labels', 'pca_n_classes', 'score'
        df_for_ylim = data_df[
            (data_df['dataset'] == dataset_name) &
            (data_df['compare_rsm_correlation'] == metric_name) &
            (data_df['region'] == region)
        ]

        initial_cond_ylim = (df_for_ylim['epoch'] == 0) & (df_for_ylim['pca_labels'] == False)
        final_cond_ylim = (df_for_ylim['epoch'] == 20) & (df_for_ylim['pca_labels'] == False) # epoch 20
        pca_cond_ylim = (df_for_ylim['epoch'] == 20) & (df_for_ylim['pca_labels'] == True) & \
                       (df_for_ylim['pca_n_classes'].isin([2, 4, 8, 16, 32, 64])) # epoch 20

        region_scores_df_for_ylim = df_for_ylim[initial_cond_ylim | final_cond_ylim | pca_cond_ylim]

        min_score_region = -0.1 # Default fallback
        max_score_region = 0.1  # Default fallback
        if not region_scores_df_for_ylim.empty and not region_scores_df_for_ylim['score'].isnull().all():
            min_score_region = region_scores_df_for_ylim['score'].min() - 0.05
            max_score_region = region_scores_df_for_ylim['score'].max() + 0.05
        else:
            print(f"Warning: No scores found for Y-limit calculation in region '{region}'. Using default limits.")

        for idx, subject in enumerate(subjects):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            # Filter data for the specific subject and region using plt_utils
            # Initial data (epoch 0)
            _, initial_data = plt_utils.split_and_select_df(
                data_df,
                dataset=dataset_name,
                metric=metric_name,
                region=region,
                epoch=0,
                subject_idx=[subject],
                reconstruct_from_pcs=False,
                layers=layer_order
            )

            initial_data = plt_utils.avg_over_seed(initial_data)

            # Final data (epoch 20, full model)
            _, final_data = plt_utils.split_and_select_df(
                data_df,
                dataset=dataset_name,
                metric=metric_name,
                region=region,
                epoch=20, # epoch 20
                subject_idx=[subject],
                reconstruct_from_pcs=False,
                layers=layer_order
            )

            final_data = plt_utils.avg_over_seed(final_data)

            # Plot PCA variants
            pca_variants = {2: 'pca2', 4: 'pca4', 8: 'pca8',
                          16: 'pca16', 32: 'pca32', 64: 'pca64'}

            for n_classes, pca_name in pca_variants.items():
                pca_subject_data, _ = plt_utils.split_and_select_df(
                    data_df,
                    dataset=dataset_name,
                    metric=metric_name,
                    region=region,
                    epoch=20, # epoch 20
                    subject_idx=[subject],
                    pca_n_classes=[n_classes],
                    reconstruct_from_pcs=False,
                    layers=layer_order
                )

                pca_subject_data = plt_utils.avg_over_seed(pca_subject_data)

                if not pca_subject_data.empty:
                    ax.plot(range(len(layer_order)), pca_subject_data.set_index('layer').reindex(layer_order)['score'],
                           color=pca_colors[n_classes], alpha=0.95, linewidth=1.5,
                           label=f'PCA-{n_classes}',
                           marker='s', markersize=4, markeredgewidth=1)

            # Plot initial and final on top
            if not initial_data.empty:
                ax.plot(range(len(layer_order)), initial_data.set_index('layer').reindex(layer_order)['score'],
                       color=colors['initial'], alpha=0.95, linewidth=2,
                       label='Initial', marker='x', markersize=5, markeredgewidth=1.5, zorder=3)

            if not final_data.empty:
                ax.plot(range(len(layer_order)), final_data.set_index('layer').reindex(layer_order)['score'],
                       color=colors['final'], alpha=0.95, linewidth=2,
                       label='Full', marker='o', markersize=5, markeredgewidth=1.5, zorder=3)

            ax.set_xticks(range(len(layer_order)))
            ax.set_xticklabels(layer_order, rotation=45, ha='right', fontsize=8)

            # Only show y-label for leftmost plots
            if col == 0:
                ax.set_ylabel('RSA Score', fontsize=10)

            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_title(f'Subject {subject + 1}', fontsize=10, pad=10)

            # Standardize y-axis limits across all plots for the current region
            ax.set_ylim(min_score_region, max_score_region)

        # Get handles and labels from the last subplot
        handles, labels = ax.get_legend_handles_labels()

        # Reorder handles and labels to match desired order
        legend_order = ['Initial', 'Full'] + [f'PCA-{n}' for n in [2, 4, 8, 16, 32, 64]]
        handles_labels = dict(zip(labels, handles))
        ordered_handles = [handles_labels[label] for label in legend_order if label in handles_labels]
        ordered_labels = [label for label in legend_order if label in handles_labels]

        # Create legend below the plots
        legend = fig.legend(ordered_handles, ordered_labels,
                          loc='center',
                          ncol=8,  # All items in one row
                          bbox_to_anchor=(0.5, 0.1),  # Moved lower
                          fontsize=12,
                          frameon=True,
                          fancybox=False,
                          edgecolor='black')

        # Add super title for the entire figure
        plt.suptitle(f'{region}', fontsize=16, y=0.95)

        # Adjust layout with specific spacing
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect to leave space for legend

        # Save plot
        os.makedirs('plotters/plots/full-vs-pcs', exist_ok=True)
        output_path = f'plotters/plots/full-vs-pcs/full-vs-pcs_individual_{region.replace(" ", "_")}.png'
        plt.savefig(output_path,
                   dpi=600,  # High DPI for sharp image
                   format='png',
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.2)  # Increased padding
        plt.close()

        print(f"Generated individual subject plot for {region} at {output_path}")

# Read the CSV file
data_df = pd.read_csv('logs/full-vs-pcs_nsd.csv') # MODIFIED: Load single CSV

# Create output directory if it doesn't exist
os.makedirs('plotters/plots/full-vs-pcs', exist_ok=True) # ENSURED full path for plots

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']

# Define dataset and metric
dataset_name = "nsd"
# Assuming 'Spearman' is the metric represented in the 'score' column
# after plt_utils.split_and_select_df (or that 'metric' is a column in the CSV)
metric_name = "Spearman"

# Create plots for all regions
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
create_individual_subject_plots(data_df, regions, layer_order, dataset_name, metric_name)
