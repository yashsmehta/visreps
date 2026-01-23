import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

def create_progression_plots(df, regions, layer_order):
    # Define colors
    colors = {
        'epoch0': '#7f8c8d',      # Gray for all epoch 0
        'regular_final': '#e74c3c',  # Red for regular training
        'pca2_final': '#2980b9',    # Blue for PCA-2
    }

    markers = {
        'regular': 'o',
        'pca2': 's'
    }

    for region in regions:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot regular training
        for epoch in [0, 20]:
            data = df[(df['pca_labels'] == False) &
                     (df['region'] == region) &
                     (df['epoch'] == epoch)]

            layer_stats = data.groupby('layer')['score'].agg(['mean', 'std']).reset_index()
            layer_stats = layer_stats.set_index('layer').loc[layer_order].reset_index()

            color = colors['epoch0'] if epoch == 0 else colors['regular_final']
            label = f'Regular Training (Epoch {epoch})'

            ax.errorbar(range(len(layer_order)),
                       layer_stats['mean'],
                       yerr=layer_stats['std'],
                       color=color,
                       alpha=0.9,
                       linewidth=2.5,
                       label=label,
                       marker=markers['regular'],
                       markersize=8,
                       markeredgewidth=2,
                       markeredgecolor='white',
                       capsize=5,
                       capthick=2)

        # Plot PCA-2 training
        for epoch in [0, 20]:
            data = df[(df['pca_labels'] == True) &
                     (df['pca_n_classes'] == 2) &
                     (df['region'] == region) &
                     (df['epoch'] == epoch)]

            layer_stats = data.groupby('layer')['score'].agg(['mean', 'std']).reset_index()
            layer_stats = layer_stats.set_index('layer').loc[layer_order].reset_index()

            color = colors['epoch0'] if epoch == 0 else colors['pca2_final']
            label = f'PCA-2 Training (Epoch {epoch})'

            ax.errorbar(range(len(layer_order)),
                       layer_stats['mean'],
                       yerr=layer_stats['std'],
                       color=color,
                       alpha=0.9,
                       linewidth=2.5,
                       label=label,
                       marker=markers['pca2'],
                       markersize=8,
                       markeredgewidth=2,
                       markeredgecolor='white',
                       capsize=5,
                       capthick=2)

        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('RSA Score', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45)

        ax.set_title(f"{region}", fontsize=16, pad=15)

        ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
        ax.set_facecolor('#f8f9fa')

        legend = ax.legend(bbox_to_anchor=(1.05, 1),
                          loc='upper left',
                          fontsize=12,
                          framealpha=0.9,
                          edgecolor='#666666',
                          fancybox=True)
        legend.get_frame().set_facecolor('#ffffff')

        plt.tight_layout()

        output_dir = 'plotters/plots/training_progression'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'training_progression_{region.replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated training progression plot for {region} at {output_path}")

# Read the CSV file
df = pd.read_csv('logs/eval/checkpoint/seeds.csv')

# Create output directory if it doesn't exist
os.makedirs('plotters/plots/training_progression', exist_ok=True)

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']

# Create plots for all regions
regions = ["ventral visual stream"]
create_progression_plots(df, regions, layer_order)
