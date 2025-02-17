import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

def create_comparison_plots(tinyimagenet_df, regions, layer_order, layer_colors):
    epochs = sorted(tinyimagenet_df[tinyimagenet_df['pca_labels'] == False]['epoch'].unique())
    
    # Create individual epoch plots for each region
    region_images = {region: [] for region in regions}
    
    # Define y-axis limits for each region
    y_limits = {
        "early visual stream": (0, 0.28),
        "midventral visual stream": (0, 0.16),
        "ventral visual stream": (0, 0.15)
    }
    
    for epoch in epochs:
        for region in regions:
            # Create figure for single region
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Filter data for the specific region and epoch
            regular_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == False) & 
                                         (tinyimagenet_df['region'] == region) & 
                                         (tinyimagenet_df['epoch'] == epoch)]
            pca_2_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                        (tinyimagenet_df['pca_n_classes'] == 2) &
                                        (tinyimagenet_df['region'] == region) & 
                                        (tinyimagenet_df['epoch'] == epoch)]
            pca_4_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == True) & 
                                        (tinyimagenet_df['pca_n_classes'] == 4) &
                                        (tinyimagenet_df['region'] == region) & 
                                        (tinyimagenet_df['epoch'] == epoch)]
            
            # Reorder data according to layer_order
            regular_ordered = regular_data.set_index('layer').loc[layer_order].reset_index()
            pca_2_ordered = pca_2_data.set_index('layer').loc[layer_order].reset_index()
            pca_4_ordered = pca_4_data.set_index('layer').loc[layer_order].reset_index()
            
            # Create the gray connecting lines (in background)
            ax.plot(range(len(layer_order)), regular_ordered['score'], 
                   color='#333333', alpha=0.4, linewidth=3, zorder=1,
                   linestyle='--')
            ax.plot(range(len(layer_order)), pca_2_ordered['score'], 
                   color='#666666', alpha=0.4, linewidth=3, zorder=1,
                   linestyle='--')
            ax.plot(range(len(layer_order)), pca_4_ordered['score'], 
                   color='#999999', alpha=0.4, linewidth=3, zorder=1,
                   linestyle='--')
            
            # Add colored points for each layer
            legend_handles = []
            for i, layer in enumerate(layer_order):
                regular_score = regular_ordered[regular_ordered['layer'] == layer]['score'].values[0]
                pca_2_score = pca_2_ordered[pca_2_ordered['layer'] == layer]['score'].values[0]
                pca_4_score = pca_4_ordered[pca_4_ordered['layer'] == layer]['score'].values[0]
                
                # Regular training points (circles)
                reg_scatter = ax.scatter(i, regular_score, 
                                       color=layer_colors[layer],
                                       s=200, zorder=2,
                                       edgecolor='white',
                                       linewidth=3,
                                       marker='o')
                
                # PCA 2-class points (triangles)
                pca_2_scatter = ax.scatter(i, pca_2_score, 
                                         color=layer_colors[layer],
                                         s=200, zorder=2,
                                         edgecolor='white',
                                         linewidth=3,
                                         marker='^')
                
                # PCA 4-class points (squares)
                pca_4_scatter = ax.scatter(i, pca_4_score, 
                                         color=layer_colors[layer],
                                         s=200, zorder=2,
                                         edgecolor='white',
                                         linewidth=3,
                                         marker='s')
                
                # Store handles for legend (only once)
                if i == 0:
                    legend_handles = [reg_scatter, pca_2_scatter, pca_4_scatter]
            
            ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
            ax.set_ylabel('RSA Score', fontsize=14, fontweight='bold')
            
            # Set y-axis limits for the region
            ax.set_ylim(*y_limits[region])
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Properly set x-axis ticks and labels
            ax.set_xticks(range(len(layer_order)))
            ax.set_xticklabels(layer_order, rotation=45)
            
            # Add title with region and epoch
            ax.set_title(f"{region}\nEpoch {epoch}", fontsize=16, pad=15)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f9fa')
            
            # Add legend
            ax.legend(legend_handles, ['Regular Training', '2-PCA Classes', '4-PCA Classes'],
                     loc='upper right', fontsize=12)
            
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = f'plots/rsa_by_epoch/temp_{region.replace(" ", "_")}_{epoch}.png'
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Read the image and append to the appropriate region list
            region_images[region].append(imageio.imread(temp_path))
            
    return region_images

# Read the CSV file
tinyimagenet_df = pd.read_csv('logs/eval/checkpoint/tiny_imagenet_cnn.csv')

# Create output directory if it doesn't exist
os.makedirs('plots/rsa_by_epoch', exist_ok=True)

# Define layer order
layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']

# Define color palettes
conv_palette = sns.color_palette("Blues", 5)
fc_palette = sns.color_palette("Reds", 3)

# Create a dictionary to map layers to colors
layer_colors = {layer: conv_palette[i] for i, layer in enumerate(layer_order[:5])}
layer_colors.update({layer: fc_palette[i] for i, layer in enumerate(layer_order[5:])})

# Create plots for all regions
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
region_images = create_comparison_plots(tinyimagenet_df, regions, layer_order, layer_colors)

# Create a GIF for each region
for region in regions:
    # Convert numpy arrays to PIL Images
    pil_images = [Image.fromarray(img) for img in region_images[region]]
    
    # Save as GIF with custom durations
    durations_ms = [d * 1000 for d in [3.0] + [0.25] * (len(pil_images) - 2) + [5.0]]
    output_path = f'plots/rsa_evolution_{region.replace(" ", "_")}.gif'
    
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=durations_ms,
        loop=0
    )
    
    print(f"Generated RSA evolution GIF for {region} at {output_path}")

# Clean up temporary files
for epoch in tinyimagenet_df['epoch'].unique():
    for region in regions:
        temp_path = f'plots/rsa_by_epoch/temp_{region.replace(" ", "_")}_{epoch}.png'
        if os.path.exists(temp_path):
            os.remove(temp_path) 