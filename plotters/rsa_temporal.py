import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set seaborn style for better aesthetics
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

def plot_layer_temporal_comparison(full_df, pca_df, layer, regions=None):
    """
    Plot temporal comparison of RSA scores for any layer across multiple regions.
    
    Args:
        full_df: DataFrame containing full model results
        pca_df: DataFrame containing PCA model results
        layer: Layer to plot (e.g., 'fc2', 'conv1', etc.)
        regions: List of brain regions to analyze
    """
    if regions is None:
        regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
    
    # Define color scheme
    colors = {
        'full': '#c0392b',  # Dark red for full model
    }
    
    # Create continuous color scale for PCA variants using blues
    n_pca_variants = 6
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants))
    pca_colors = dict(zip([2, 4, 8, 16, 32, 64], blues))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"{layer.upper()} Layer Brain Score Over ImageNet Training", 
                fontsize=15, y=0.95, fontweight='medium')
    
    for idx, region in enumerate(regions):
        ax = axes[idx]
        
        # Filter for specified region and layer
        region_full_df = full_df[(full_df['region'] == region) & (full_df['layer'] == layer)]
        
        # Average across subject_idx
        region_full_df = region_full_df.groupby(['epoch', 'layer', 'region'])['score'].mean().reset_index()
        region_pca_df = pca_df.groupby(['epoch', 'layer', 'region', 'pca_labels', 'pca_n_classes'])['score'].mean().reset_index()
        
        # Plot full model line
        ax.plot(region_full_df['epoch'], region_full_df['score'],
                color=colors['full'], alpha=0.95, linewidth=2.5,
                label='Full',
                marker='o', markersize=7, markeredgewidth=1.5,
                zorder=3)
        
        # Plot PCA variants
        pca_variants = [2, 4, 8, 16, 32, 64]
        for n_classes in pca_variants:
            layer_pca = region_pca_df[(region_pca_df['pca_labels'] == True) &
                              (region_pca_df['pca_n_classes'] == n_classes) &
                              (region_pca_df['region'] == region) &
                              (region_pca_df['layer'] == layer)]
            
            ax.plot(layer_pca['epoch'], layer_pca['score'],
                    color=pca_colors[n_classes], alpha=0.95, linewidth=2.5,
                    label=f'PCA-{n_classes}',
                    marker='s', markersize=7, markeredgewidth=1.5,
                    zorder=2)
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='medium', labelpad=10)
        if idx == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('RSA Score', fontsize=14, fontweight='medium', labelpad=10)
        
        ax.tick_params(axis='both', which='major', labelsize=12, pad=8, length=6)
        ax.set_xticks(range(0, 11))
        
        # Add subtitle for each region
        ax.set_title(f"{region.title()}", fontsize=14, pad=15)
        
        # Enhanced grid
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666', zorder=1)
        ax.set_facecolor('white')
        
        # Only add legend to the last subplot
        if idx == 2:
            legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0)
    
    fig.patch.set_facecolor('white')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = f'plots/rsa_temporal/{layer}_temporal_all_regions.png'
    os.makedirs('plots/rsa_temporal', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Generated {layer.upper()} temporal plot for all regions at {output_path}")

# Read the CSV files
full_model_df = pd.read_csv('logs/eval/checkpoint/imagenet_cnn.csv')
pca_df = pd.read_csv('logs/eval/checkpoint/imagenet_pca.csv')

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Set layer to plot
layer = 'conv5'

# Generate plot for all regions
plot_layer_temporal_comparison(full_model_df, pca_df, layer) 