import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

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

def create_individual_subject_plots(full_df, pca_df, regions, layer_order):
    # Define color scheme
    colors = {
        'initial': '#7f8c8d',    # Elegant gray
        'final': '#c0392b',      # Dark red/orange for full model
    }
    
    # Create continuous color scale for PCA variants using blues
    n_pca_variants = 6
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants))
    pca_colors = dict(zip([2, 4, 8, 16, 32, 64], blues))
    
    # Get unique subjects
    subjects = sorted(full_df['subject_idx'].unique())
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
        
        for idx, subject in enumerate(subjects):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Filter data for the specific subject and region
            subject_full = full_df[(full_df['subject_idx'] == subject) & 
                                 (full_df['region'] == region)]
            
            # Plot initial and final epochs
            initial_data = subject_full[subject_full['epoch'] == 0]
            final_data = subject_full[subject_full['epoch'] == 10]
            
            # Plot PCA variants
            pca_variants = {2: 'pca2', 4: 'pca4', 8: 'pca8', 
                          16: 'pca16', 32: 'pca32', 64: 'pca64'}
            
            for n_classes, pca_name in pca_variants.items():
                pca_subject = pca_df[(pca_df['subject_idx'] == subject) & 
                                   (pca_df['region'] == region) &
                                   (pca_df['pca_labels'] == True) &
                                   (pca_df['pca_n_classes'] == n_classes) &
                                   (pca_df['epoch'] == 10)]
                
                if not pca_subject.empty:
                    ax.plot(range(len(layer_order)), pca_subject.set_index('layer').reindex(layer_order)['score'],
                           color=pca_colors[n_classes], alpha=0.95, linewidth=1.5,
                           label=f'PCA-{n_classes}',
                           marker='s', markersize=4, markeredgewidth=1)
            
            # Plot initial and final on top
            ax.plot(range(len(layer_order)), initial_data.set_index('layer').reindex(layer_order)['score'],
                   color=colors['initial'], alpha=0.95, linewidth=2,
                   label='Initial', marker='x', markersize=5, markeredgewidth=1.5, zorder=3)
            
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
            
            # Standardize y-axis limits across all plots
            ax.set_ylim(full_df[full_df['region'] == region]['score'].min() - 0.05,
                       full_df[full_df['region'] == region]['score'].max() + 0.05)
        
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
        output_path = f'plots/full-vs-pcs/full-vs-pcs_individual_{region.replace(" ", "_")}.png'
        plt.savefig(output_path, 
                   dpi=600,  # High DPI for sharp image
                   format='png',
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.2)  # Increased padding
        plt.close()
        
        print(f"Generated individual subject plot for {region} at {output_path}")

# Read the CSV files
full_model_df = pd.read_csv('logs/eval/checkpoint/imagenet_cnn.csv')
pca_df = pd.read_csv('logs/eval/checkpoint/imagenet_pca.csv')

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']

# Create plots for all regions
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
create_individual_subject_plots(full_model_df, pca_df, regions, layer_order) 