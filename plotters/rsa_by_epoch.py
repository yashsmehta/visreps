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

def create_rsa_plots(df, region, layer_order, layer_colors):
    # Filter data for the specific region
    region_df = df[(df['pca_labels'] == False) & (df['region'] == region)]
    epochs = sorted(region_df['epoch'].unique())
    
    # Create individual epoch line plots
    images = []
    for epoch in epochs:
        # Fix: Use region_df instead of df for epoch filtering
        epoch_data = region_df[region_df['epoch'] == epoch]
        
        plt.figure(figsize=(8, 7))
        
        # Reorder data according to layer_order for consistent plotting
        ordered_data = epoch_data.set_index('layer').loc[layer_order].reset_index()
        
        # Create the gray connecting line first (in background)
        plt.plot(ordered_data['layer'], ordered_data['score'], 
                color='gray', alpha=0.5, linewidth=3, zorder=1,
                linestyle='--')
        
        # Add colored points for each layer
        for layer in layer_order:
            score = ordered_data[ordered_data['layer'] == layer]['score'].values[0]
            plt.scatter(layer, score, 
                       color=layer_colors[layer],
                       s=175,  # larger point size
                       zorder=2,
                       edgecolor='white',
                       linewidth=2,
                       label=layer)
        
        plt.xlabel('Layer', fontsize=14, fontweight='bold')
        plt.ylabel('RSA Score', fontsize=14, fontweight='bold')
        plt.ylim(0, 0.32)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Format region name to be title case and remove "visual stream"
        region_title = region.replace(" visual stream", "").title()
        plt.title(f"{region_title}\nEpoch {epoch}", fontsize=16, fontweight='bold', pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a light gray background
        plt.gca().set_facecolor('#f8f9fa')
        plt.grid(True, color='white', linestyle='-', linewidth=1.5, zorder=0)
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = f'plots/rsa_by_epoch/temp_{region}_{epoch}.png'
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Read the image
        images.append(imageio.imread(temp_path))
        
    return images

# Read the CSV file
df = pd.read_csv('logs/eval/checkpoint/imagenet_cnn.csv')

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

# Create plots for each region
regions = ["early visual stream", "midventral visual stream", "ventral visual stream"]
region_images = {region: create_rsa_plots(df, region, layer_order, layer_colors) 
                for region in regions}

# Combine images horizontally for the GIF
combined_images = []
for i in range(len(region_images[regions[0]])):  # Assume all regions have same number of epochs
    frames = [region_images[region][i] for region in regions]
    # Combine horizontally with a small white border
    combined = np.hstack([np.pad(frame, ((10, 10), (10, 10), (0, 0)), 'constant', constant_values=255) 
                         for frame in frames])
    combined_images.append(combined)

# Convert numpy arrays to PIL Images
pil_images = [Image.fromarray(img) for img in combined_images]

# Save as GIF with custom durations (durations must be in milliseconds for PIL)
durations_ms = [d * 1000 for d in [3.0] + [0.5] * (len(combined_images) - 2) + [1.0]]  # First frame 2s, middle frames 0.5s, last frame 1s
pil_images[0].save(
    'plots/rsa_evolution_imagenet.gif',
    save_all=True,
    append_images=pil_images[1:],
    duration=durations_ms,
    loop=0
)

# Clean up temporary files
for region in regions:
    for epoch in df['epoch'].unique():
        temp_path = f'plots/rsa_by_epoch/temp_{region}_{epoch}.png'
        if os.path.exists(temp_path):
            os.remove(temp_path)

print("Generated combined RSA evolution GIF at plots/rsa_evolution_imagenet.gif") 