import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

# Set seaborn style to match full-vs-pcs.py
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
    # Print statistics about the dataframe
    print("\n===== DataFrame Statistics =====")
    print(f"Total rows: {len(df)}")
    print(f"Unique epochs: {sorted(df['epoch'].unique())}")
    print(f"Regions: {sorted(df['region'].unique())}")
    print(f"PCA variants: {sorted(df['pca_n_classes'].unique())}")
    
    # Count by region
    region_counts = df['region'].value_counts().sort_index()
    print("\nCount by region:")
    for region, count in region_counts.items():
        print(f"  {region}: {count}")
    
    # Count by epoch
    epoch_counts = df['epoch'].value_counts().sort_index()
    print("\nCount by epoch:")
    for epoch, count in epoch_counts.items():
        print(f"  {epoch}: {count}")
        
    # Count by PCA variant
    pca_counts = df['pca_n_classes'].value_counts().sort_index()
    print("\nCount by PCA variant:")
    for pca_n, count in pca_counts.items():
        print(f"  {pca_n}: {count}")
    
    # Count by layer
    layer_counts = df['layer'].value_counts()
    print("\nCount by layer:")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count}")
    
    print("\n===============================")
    
    # Color scheme for different PCA variants - using Blues colormap to match full-vs-pcs.py
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, 6))  # From light to dark blue
    color_dict = dict(zip([2, 4, 8, 16, 32, 64], blues))
    
    # Create plots for each region
    for region in regions:
        print(f"\nProcessing region: {region}")
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # First plot epoch 0 in gray (average across all PCA variants)
        epoch0_data = df[(df['region'] == region) & (df['epoch'] == 0)]
        
        # Calculate mean across all PCA variants for each layer
        epoch0_layer_stats = []
        for layer in layer_order:
            layer_scores = epoch0_data[epoch0_data['layer'] == layer]['score']
            mean = layer_scores.mean()
            epoch0_layer_stats.append({'layer': layer, 'mean': mean})
        
        epoch0_layer_stats_df = pd.DataFrame(epoch0_layer_stats)
        
        # Plot epoch 0 in gray - no error bars since there's no spread across seeds
        # Using 'x' marker to match full-vs-pcs.py initial style
        ax.plot(range(len(layer_order)), 
                epoch0_layer_stats_df['mean'],
                label='Initial',
                color='#7f8c8d',  # Match the 'initial' color from full-vs-pcs.py
                marker='x', 
                markersize=8,
                markeredgewidth=2,
                linewidth=2.5,
                alpha=0.95,
                zorder=3)
        
        # Get data for epoch 20 for this region
        region_data = df[(df['region'] == region) & (df['epoch'] == 20)]
        
        # Plot lines for each PCA variant
        pca_handles = []  # Store handles for custom legend
        pca_labels = []   # Store labels for custom legend
        
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
            line = ax.errorbar(range(len(layer_order)), 
                       layer_stats_df['mean'],
                       yerr=layer_stats_df['std'],
                       label=str(pca_n),
                       color=color_dict[pca_n],
                       marker='s',  # Square marker to match full-vs-pcs.py
                       markersize=7,
                       markeredgewidth=1.5,
                       capsize=4,
                       capthick=1.5,
                       linewidth=2.5,
                       alpha=0.95,
                       zorder=2)
            
            pca_handles.append(line)
            pca_labels.append(str(pca_n))
        
        # Customize plot
        ax.set_xlabel('Layer', fontsize=14, fontweight='medium', labelpad=10)
        ax.set_ylabel('RSA Score', fontsize=14, fontweight='medium', labelpad=10)
        ax.set_title(f"{region}", fontsize=16, pad=20, fontweight='medium')
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45, ha='right')
        
        # Improve tick parameters to match full-vs-pcs.py
        ax.tick_params(axis='both', which='major', labelsize=12, pad=8, length=6)
        
        # Add grid and set background color
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666', zorder=1)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Create custom legend with specific ordering like in full-vs-pcs.py
        handles, labels = ax.get_legend_handles_labels()
        
        # First legend for Initial
        initial_handle = handles[0]
        initial_label = labels[0]
        legend = ax.legend([initial_handle], [initial_label], 
                          bbox_to_anchor=(1.02, 1), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=1)
        
        # Second legend for first row of PCA numbers (2,4,8)
        legend2 = ax.legend(handles[1:4], labels[1:4],
                          bbox_to_anchor=(1.02, 0.85), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=3)
        
        # Third legend for second row of PCA numbers (16,32,64)
        legend3 = ax.legend(handles[4:], labels[4:],
                          bbox_to_anchor=(1.02, 0.8), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=3)
        
        # Add back all legends in reverse order (bottom to top)
        ax.add_artist(legend2)
        ax.add_artist(legend)
        
        # Adjust layout
        plt.tight_layout()
        
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
# regions = ["early visual stream"]

# Create plots
create_rsa_plots(df, regions, layer_order)
