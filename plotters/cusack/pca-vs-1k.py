"""
Plot brain alignment scores for EVC region across different class granularities.
Shows best layer performance for PCA models (2-64 classes) vs standard 1000-way.
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
PCA_CLASSES = [2, 4, 8, 16, 32, 64]
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

# Create color scheme
blues = plt.cm.Blues(np.linspace(0.4, 0.85, len(PCA_CLASSES)))
orange = '#FF8C00'

# Create figure with shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

for idx, age_group in enumerate(AGE_GROUPS):
    ax = axes[idx]
    
    n_classes_list = []
    best_scores = []
    best_layers = []
    colors = []
    
    # Get best scores for each PCA model
    for i, n_classes in enumerate(PCA_CLASSES):
        df_filtered = df_pca[
            (df_pca["region"] == REGION) &
            (df_pca["age_group"] == age_group) &
            (df_pca["pca_n_classes"] == n_classes)
        ]
        
        if not df_filtered.empty:
            best_score = df_filtered["score"].max()
            best_layer = df_filtered.loc[df_filtered['score'].idxmax(), 'layer']
            n_classes_list.append(n_classes)
            best_scores.append(best_score)
            best_layers.append(best_layer)
            colors.append(blues[i])
    
    # Get best score for 1000-way model
    df_1000 = df_1k[
        (df_1k["region"] == REGION) &
        (df_1k["age_group"] == age_group)
    ]
    
    if not df_1000.empty:
        best_score_1000 = df_1000["score"].max()
        best_layer_1000 = df_1000.loc[df_1000['score'].idxmax(), 'layer']
        n_classes_list.append(1000)
        best_scores.append(best_score_1000)
        best_layers.append(best_layer_1000)
        colors.append(orange)
    
    # Plot bars
    x_pos = np.arange(len(n_classes_list))
    bars = ax.bar(x_pos, best_scores, color=colors, width=0.65, edgecolor='black', linewidth=0.8)
    
    # Add layer labels at bottom of bars
    for bar, layer, score in zip(bars, best_layers, best_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 0.02,
                layer, ha='center', va='bottom', fontsize=6.5, fontweight='bold',
                color='white', rotation=0)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(n_classes_list, fontsize=9)
    ax.set_xlabel("Number of classes", fontsize=11, fontweight='bold')
    
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
output_file = f"plotters/cusack/{REGION.lower()}_alignment.png"
plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✓ Saved plot to {output_file}")
plt.show()