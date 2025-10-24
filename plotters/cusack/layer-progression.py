"""
Plot brain alignment scores across all network layers.
Shows how RSA scores progress through the network layers for 1000-way classification.
Publication-quality figure for Nature journal submission.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Configuration
CSV_1K = "logs/cusack_1k.csv"
CSV_PCA = "logs/cusack_pca.csv"
REGION = "vvc"  # Change this to plot different regions: "evc", "vvc", "V1", etc.
PCA_N_CLASSES = 64  # PCA model to compare
LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
AGE_GROUPS = ["2month", "9month"]

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0

# Load data
df_1k = pd.read_csv(CSV_1K)
df_pca = pd.read_csv(CSV_PCA)

# Color scheme
blue = '#4472C4'
orange = '#FF8C00'

# Create figure with shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for idx, age_group in enumerate(AGE_GROUPS):
    ax = axes[idx]
    
    # Get scores for all layers - PCA 64-class
    layer_scores_pca = []
    for layer in LAYERS:
        df_filtered = df_pca[
            (df_pca["region"] == REGION) &
            (df_pca["age_group"] == age_group) &
            (df_pca["pca_n_classes"] == PCA_N_CLASSES) &
            (df_pca["layer"] == layer)
        ]
        
        if not df_filtered.empty:
            score = df_filtered["score"].values[0]
            layer_scores_pca.append(score)
        else:
            layer_scores_pca.append(np.nan)
    
    # Get scores for all layers - 1000-way
    layer_scores_1k = []
    for layer in LAYERS:
        df_filtered = df_1k[
            (df_1k["region"] == REGION) &
            (df_1k["age_group"] == age_group) &
            (df_1k["layer"] == layer)
        ]
        
        if not df_filtered.empty:
            score = df_filtered["score"].values[0]
            layer_scores_1k.append(score)
        else:
            layer_scores_1k.append(np.nan)
    
    # Plot lines
    x_pos = np.arange(len(LAYERS))
    ax.plot(x_pos, layer_scores_pca, color=blue, linewidth=2.5, 
            marker='o', markersize=8, markeredgecolor='black', 
            markeredgewidth=1.0, label=f'{PCA_N_CLASSES}-class PCA')
    ax.plot(x_pos, layer_scores_1k, color=orange, linewidth=2.5, 
            marker='o', markersize=8, markeredgecolor='black', 
            markeredgewidth=1.0, label='1000-way')
    
    # Add legend
    ax.legend(loc='best', fontsize=9, frameon=False)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LAYERS, fontsize=9)
    ax.set_xlabel("Network layer", fontsize=11, fontweight='bold')
    
    if idx == 0:
        ax.set_ylabel("RSA", fontsize=11, fontweight='bold')
    
    # Title formatting
    age_label = age_group.replace('month', ' months')
    ax.set_title(age_label, fontsize=12, fontweight='bold', pad=10)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

# Add main title
fig.suptitle(REGION.upper(), fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
output_file = f"plotters/cusack/{REGION.lower()}_layer_progression.png"
plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Saved plot to {output_file}")
plt.show()

