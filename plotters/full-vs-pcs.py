import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

def create_comparison_plots(tinyimagenet_df, regions, layer_order, layer_colors):
    # Define a modern color palette
    colors = {
        'initial': '#7f8c8d',    # Elegant gray
        'final': '#f39c12',      # Warm orange
        'pca2': '#2980b9',       # Darker blue
        'pca4': '#3498db',       # Medium blue
        'pca8': '#85c1e9',       # Light blue
        'pca16': '#bde3fc'       # Lightest blue
    }
    
    # Create individual plots for each region
    for region in regions:
        # Create figure for single region with extra width for legend
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter data for the specific region and epochs
        initial_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == False) & 
                                     (tinyimagenet_df['region'] == region) & 
                                     (tinyimagenet_df['epoch'] == 0)]
        
        final_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == False) & 
                                    (tinyimagenet_df['region'] == region) & 
                                    (tinyimagenet_df['epoch'] == 20)]
        
        pca2_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                   (tinyimagenet_df['pca_n_classes'] == 2) &
                                   (tinyimagenet_df['region'] == region) & 
                                   (tinyimagenet_df['epoch'] == 20)]
        
        pca4_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                   (tinyimagenet_df['pca_n_classes'] == 4) &
                                   (tinyimagenet_df['region'] == region) & 
                                   (tinyimagenet_df['epoch'] == 20)]

        pca8_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                   (tinyimagenet_df['pca_n_classes'] == 8) &
                                   (tinyimagenet_df['region'] == region) & 
                                   (tinyimagenet_df['epoch'] == 20)]

        pca16_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                    (tinyimagenet_df['pca_n_classes'] == 16) &
                                    (tinyimagenet_df['region'] == region) & 
                                    (tinyimagenet_df['epoch'] == 20)]
        
        # Reorder data according to layer_order
        initial_ordered = initial_data.set_index('layer').loc[layer_order].reset_index()
        final_ordered = final_data.set_index('layer').loc[layer_order].reset_index()
        pca2_ordered = pca2_data.set_index('layer').loc[layer_order].reset_index()
        pca4_ordered = pca4_data.set_index('layer').loc[layer_order].reset_index()
        pca8_ordered = pca8_data.set_index('layer').loc[layer_order].reset_index()
        pca16_ordered = pca16_data.set_index('layer').loc[layer_order].reset_index()
        
        # Plot lines with enhanced styling
        ax.plot(range(len(layer_order)), initial_ordered['score'], 
               color=colors['initial'], alpha=0.9, linewidth=2.5, 
               label='Initial (Epoch 0)',
               marker='o', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')
        
        ax.plot(range(len(layer_order)), final_ordered['score'], 
               color=colors['final'], alpha=0.9, linewidth=2.5, 
               label='Final (Epoch 20)',
               marker='s', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')
        
        ax.plot(range(len(layer_order)), pca2_ordered['score'], 
               color=colors['pca2'], alpha=0.9, linewidth=2.5, 
               label='PCA-2 (Epoch 20)',
               marker='^', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')
        
        ax.plot(range(len(layer_order)), pca4_ordered['score'], 
               color=colors['pca4'], alpha=0.9, linewidth=2.5, 
               label='PCA-4 (Epoch 20)',
               marker='D', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')
        
        ax.plot(range(len(layer_order)), pca8_ordered['score'], 
               color=colors['pca8'], alpha=0.9, linewidth=2.5, 
               label='PCA-8 (Epoch 20)',
               marker='v', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')

        ax.plot(range(len(layer_order)), pca16_ordered['score'], 
               color=colors['pca16'], alpha=0.9, linewidth=2.5, 
               label='PCA-16 (Epoch 20)',
               marker='p', markersize=8, markeredgewidth=2, 
               markeredgecolor='white')
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('RSA Score', fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45)
        
        # Add title with region
        ax.set_title(f"{region}", fontsize=16, pad=15)
        
        # Enhanced grid and background
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
        ax.set_facecolor('#f8f9fa')
        
        # Add legend outside the plot
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=12, framealpha=0.9,
                          edgecolor='#666666', fancybox=True)
        legend.get_frame().set_facecolor('#ffffff')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save plot with extra space for legend
        output_path = f'plots/full-vs-pcs_{region.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated RSA comparison plot for {region} at {output_path}")

# Read the CSV file
tinyimagenet_df = pd.read_csv('logs/eval/checkpoint/tiny_custom_cnn.csv')

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']

# Create plots for all regions
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
create_comparison_plots(tinyimagenet_df, regions, layer_order, None) 