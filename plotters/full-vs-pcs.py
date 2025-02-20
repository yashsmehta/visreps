import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors

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

def create_comparison_plots(full_df, pca_df, regions, layer_order):
    # Define color scheme
    colors = {
        'initial': '#7f8c8d',    # Elegant gray
        'final': '#c0392b',      # Dark red/orange for full model
    }
    
    # Create continuous color scale for PCA variants using blues
    n_pca_variants = 6
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants))  # From light to dark blue
    pca_colors = dict(zip([2, 4, 8, 16, 32, 64], blues))
    
    # Average across subjects for both dataframes
    group_cols = ['region', 'layer', 'epoch']  # Columns to group by for full model
    
    # Print number of subjects for full model
    n_subjects_full = len(full_df['subject_idx'].unique())
    print(f"\nAveraging over {n_subjects_full} subjects for full model")
    
    full_df = full_df.groupby(group_cols)['score'].mean().reset_index()
    
    pca_group_cols = ['region', 'layer', 'epoch', 'pca_labels', 'pca_n_classes']  # Columns to group by for PCA
    
    # Print number of subjects for PCA model
    n_subjects_pca = len(pca_df['subject_idx'].unique())
    print(f"Averaging over {n_subjects_pca} subjects for PCA variants")
    
    pca_df = pca_df.groupby(pca_group_cols)['score'].mean().reset_index()
    
    # Create individual plots for each region
    for region in regions:
        print(f"\nProcessing region: {region}")
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Filter data for the specific region and epochs - Full model data
        initial_data = full_df[(full_df['region'] == region) & 
                             (full_df['epoch'] == 0)]
        
        final_data = full_df[(full_df['region'] == region) & 
                           (full_df['epoch'] == 10)]
        
        # Filter data for PCA variants
        pca_variants = {
            'pca2': 2,
            'pca4': 4,
            'pca8': 8,
            'pca16': 16,
            'pca32': 32,
            'pca64': 64
        }
        
        pca_data = {}
        for pca_name, n_classes in pca_variants.items():
            pca_data[pca_name] = pca_df[(pca_df['pca_labels'] == True) & 
                                       (pca_df['pca_n_classes'] == n_classes) &
                                       (pca_df['region'] == region) & 
                                       (pca_df['epoch'] == 10)]
        
        # Reorder data according to layer_order
        initial_ordered = initial_data.set_index('layer').reindex(layer_order).reset_index()
        final_ordered = final_data.set_index('layer').reindex(layer_order).reset_index()
        
        pca_ordered = {}
        for pca_name in pca_variants.keys():
            pca_ordered[pca_name] = pca_data[pca_name].set_index('layer').reindex(layer_order).reset_index()
        
        # Plot lines with enhanced styling
        ax.plot(range(len(layer_order)), initial_ordered['score'], 
               color=colors['initial'], alpha=0.95, linewidth=2.5, 
               label='Initial',
               marker='x', markersize=8, markeredgewidth=2, 
               zorder=3)
        
        ax.plot(range(len(layer_order)), final_ordered['score'], 
               color=colors['final'], alpha=0.95, linewidth=2.5, 
               label='Full',
               marker='o', markersize=7, markeredgewidth=1.5, 
               zorder=3)
        
        # Plot all PCA variants in order
        for pca_name in ['pca2', 'pca4', 'pca8', 'pca16', 'pca32', 'pca64']:
            n_classes = pca_variants[pca_name]
            ax.plot(range(len(layer_order)), pca_ordered[pca_name]['score'], 
                   color=pca_colors[n_classes], alpha=0.95, linewidth=2.5, 
                   label=str(n_classes),
                   marker='s', markersize=7, markeredgewidth=1.5, 
                   zorder=2)
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='medium', labelpad=10)
        ax.set_ylabel('RSA Score', fontsize=14, fontweight='medium', labelpad=10)
        
        ax.tick_params(axis='both', which='major', labelsize=12, pad=8, length=6)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45, ha='right')
        
        # Add title with region
        ax.set_title(f"{region}", fontsize=16, pad=20, fontweight='medium')
        
        # Enhanced grid
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666', zorder=1)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Create custom legend with specific ordering
        handles, labels = ax.get_legend_handles_labels()
        
        # First legend for Initial and Full
        legend = ax.legend(handles[:2], labels[:2], 
                          bbox_to_anchor=(1.02, 1), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=1)
        
        # Second legend for first row of PCA numbers (2,4,8)
        legend2 = ax.legend(handles[2:5], labels[2:5],
                          bbox_to_anchor=(1.02, 0.85), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=3)
        
        # Third legend for second row of PCA numbers (16,32,64)
        legend3 = ax.legend(handles[5:], labels[5:],
                          bbox_to_anchor=(1.02, 0.8), loc='upper left',
                          fontsize=11, framealpha=1.0,
                          edgecolor='none', fancybox=False,
                          borderaxespad=0,
                          ncol=3)
        
        # Add back all legends in reverse order (bottom to top)
        ax.add_artist(legend2)
        ax.add_artist(legend)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save plot with extra space for legend
        output_path = f'plots/full-vs-pcs_{region.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated RSA comparison plot for {region} at {output_path}")

# Read the CSV files
full_model_df = pd.read_csv('logs/eval/checkpoint/imagenet_cnn.csv')
pca_df = pd.read_csv('logs/eval/checkpoint/imagenet_pca.csv')

# Print unique regions from both dataframes
print("\nAvailable regions in full model:", full_model_df['region'].unique())
print("Available regions in PCA model:", pca_df['region'].unique())

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']

# Create plots for all regions
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
create_comparison_plots(full_model_df, pca_df, regions, layer_order) 