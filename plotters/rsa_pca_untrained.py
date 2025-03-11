import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set seaborn style
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

def create_rsa_plots(df, regions, layer_order):
    # Color scheme for different PCA variants
    pca_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))  # 6 colors for PCA variants
    color_dict = dict(zip([2, 4, 8, 16, 32, 64], pca_colors))
    
    # Create plots for each region
    for region in regions:
        print(f"\nProcessing region: {region}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data for epoch 20 for this region
        region_data = df[(df['region'] == region) & (df['epoch'] == 20)]
        
        # Plot lines for each PCA variant
        for pca_n in [2, 4, 8, 16, 32, 64]:
            pca_data = region_data[region_data['pca_n_classes'] == pca_n]
            
            # Calculate mean and std across seeds for each layer
            layer_stats = []
            for layer in layer_order:
                layer_scores = pca_data[pca_data['layer'] == layer]['score']
                mean = layer_scores.mean()
                std = layer_scores.std()
                layer_stats.append({'layer': layer, 'mean': mean, 'std': std})
            
            layer_stats_df = pd.DataFrame(layer_stats)
            
            # Plot mean line with error bars
            ax.errorbar(range(len(layer_order)), 
                       layer_stats_df['mean'],
                       yerr=layer_stats_df['std'],
                       label=f'PCA-{pca_n}',
                       color=color_dict[pca_n],
                       marker='o',
                       markersize=6,
                       capsize=4,
                       capthick=1.5,
                       linewidth=2,
                       alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Layer', fontsize=14, fontweight='medium', labelpad=10)
        ax.set_ylabel('RSA Score', fontsize=14, fontweight='medium', labelpad=10)
        ax.set_title(f"{region}", fontsize=16, pad=20, fontweight='medium')
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45, ha='right')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666', zorder=1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        output_path = f'plots/rsa_pca_untrained_{region.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated RSA plot for {region} at {output_path}")

# Read the data
df = pd.read_csv('logs/eval/checkpoint/imagenet_pca_untrained.csv')

# Define layer order and regions
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]

# Create plots
create_rsa_plots(df, regions, layer_order)
