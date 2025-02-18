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

def create_comparison_plots(imagenet_df, tinyimagenet_df, regions, layer_order, layer_colors):
    epochs = sorted(imagenet_df[imagenet_df['pca_labels'] == False]['epoch'].unique())
    
    # Create individual epoch plots
    images = []
    for epoch in epochs:
        # Create figure with three subplots sharing y-axis
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        fig.subplots_adjust(wspace=0.05)  # Reduce space between subplots
        
        # Store handles for legend
        legend_handles = []
        
        for idx, region in enumerate(regions):
            ax = axes[idx]
            
            # Filter data for the specific region and epoch
            imagenet_data = imagenet_df[(imagenet_df['pca_labels'] == False) & 
                                      (imagenet_df['region'] == region) & 
                                      (imagenet_df['epoch'] == epoch)]
            tinyimagenet_data = tinyimagenet_df[(tinyimagenet_df['pca_labels'] == False) & 
                                               (tinyimagenet_df['region'] == region) & 
                                               (tinyimagenet_df['epoch'] == epoch)]
            
            # Reorder data according to layer_order
            imagenet_ordered = imagenet_data.set_index('layer').loc[layer_order].reset_index()
            tinyimagenet_ordered = tinyimagenet_data.set_index('layer').loc[layer_order].reset_index()
            
            # Create the gray connecting lines (in background)
            ax.plot(range(len(layer_order)), imagenet_ordered['score'], 
                   color='gray', alpha=0.3, linewidth=2, zorder=1,
                   linestyle='--')
            ax.plot(range(len(layer_order)), tinyimagenet_ordered['score'], 
                   color='gray', alpha=0.3, linewidth=2, zorder=1,
                   linestyle='--')
            
            # Add colored points for each layer
            for i, layer in enumerate(layer_order):
                imagenet_score = imagenet_ordered[imagenet_ordered['layer'] == layer]['score'].values[0]
                tinyimagenet_score = tinyimagenet_ordered[tinyimagenet_ordered['layer'] == layer]['score'].values[0]
                
                # ImageNet points (circles)
                im_scatter = ax.scatter(i, imagenet_score, 
                                      color=layer_colors[layer],
                                      s=175, zorder=2,
                                      edgecolor='white',
                                      linewidth=2,
                                      marker='o')
                
                # Tiny-ImageNet points (triangles)
                tiny_scatter = ax.scatter(i, tinyimagenet_score, 
                                        color=layer_colors[layer],
                                        s=175, zorder=2,
                                        edgecolor='white',
                                        linewidth=2,
                                        marker='^')
                
                # Store handles for first subplot only
                if idx == 0 and i == 0:
                    legend_handles = [im_scatter, tiny_scatter]
            
            ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('RSA Score', fontsize=14, fontweight='bold')
            
            ax.set_ylim(0, 0.32)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Properly set x-axis ticks and labels
            ax.set_xticks(range(len(layer_order)))
            ax.set_xticklabels(layer_order, rotation=45)
            
            # Format region name to be title case and remove "visual stream"
            region_title = region
            ax.set_title(f"{region_title}", fontsize=16, pad=15)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f9fa')
            
        # Add epoch number as overall suptitle
        plt.suptitle(f"Epoch {epoch}", fontsize=18, fontweight='bold', y=1.05)
        
        # Add common legend outside the plots
        fig.legend(legend_handles, ['ImageNet', 'Tiny-ImageNet'],
                  loc='center right', bbox_to_anchor=(1.1, 0.5),
                  fontsize=12)
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = f'plots/rsa_by_epoch/temp_epoch_{epoch}.png'
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Read the image
        images.append(imageio.imread(temp_path))
        
    return images

# Read the CSV files
imagenet_df = pd.read_csv('logs/eval/checkpoint/imagenet_cnn.csv')
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
images = create_comparison_plots(imagenet_df, tinyimagenet_df, regions, layer_order, layer_colors)

# Convert numpy arrays to PIL Images
pil_images = [Image.fromarray(img) for img in images]

# Save as GIF with custom durations (durations must be in milliseconds for PIL)
durations_ms = [d * 1000 for d in [3.0] + [0.5] * (len(images) - 2) + [1.0]]  # First frame 3s, middle frames 0.5s, last frame 1s
pil_images[0].save(
    'plots/rsa_evolution_comparison.gif',
    save_all=True,
    append_images=pil_images[1:],
    duration=durations_ms,
    loop=0
)

# Clean up temporary files
for epoch in imagenet_df['epoch'].unique():
    temp_path = f'plots/rsa_by_epoch/temp_epoch_{epoch}.png'
    if os.path.exists(temp_path):
        os.remove(temp_path)

print("Generated combined RSA evolution GIF at plots/rsa_evolution_comparison.gif") 