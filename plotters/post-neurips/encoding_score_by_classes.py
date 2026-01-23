"""
Encoding score vs number of coarse-grained classes.
Two subplots: Early visual stream (conv4) and Ventral visual stream (fc2).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Nature-style plot settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Paths
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
OUTPUT_DIR = Path(__file__).parent

# Load data
df_pca = pd.read_csv(LOGS_DIR / "encoding_score.csv")
df_1k = pd.read_csv(LOGS_DIR / "encoding_score_1k.csv")

# Config
LAYER_MAP = {
    "early visual stream": "conv4",
    "ventral visual stream": "fc2",
}
SUBJECTS = [0, 1, 3]

# Aggregate: mean across subjects
def get_mean(df, region, layer):
    """Get mean for each pca_n_classes."""
    mask = (df["region"] == region) & (df["layer"] == layer) & (df["subject_idx"].isin(SUBJECTS))
    return df[mask].groupby("pca_n_classes")["score"].mean()

def get_baseline(df, region, layer):
    """Get baseline (1000-class) mean."""
    mask = (df["region"] == region) & (df["layer"] == layer) & (df["subject_idx"].isin(SUBJECTS))
    return df[mask]["score"].mean()

# Create figure (Nature single column width ~89mm, double ~183mm)
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)

regions = ["early visual stream", "ventral visual stream"]
titles = ["Early Visual Stream", "Ventral Visual Stream"]

# 6 shades of blue (light to dark)
blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
baseline_color = "#e31a1c"

for ax, region, title in zip(axes, regions, titles):
    layer = LAYER_MAP[region]
    
    # PCA data
    mean = get_mean(df_pca, region, layer)
    x = np.arange(len(mean))
    labels = mean.index.values
    
    # Bar plot with gradient blues
    bars = ax.bar(x, mean.values, color=blue_shades, edgecolor='white', linewidth=0.5, zorder=3)
    
    # Baseline
    baseline_mean = get_baseline(df_1k, region, layer)
    baseline_line = ax.axhline(baseline_mean, color=baseline_color, linestyle='--', 
                                linewidth=1.5, zorder=4)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Encoding Score")
    ax.set_title(f"{title} ({layer})")
    
    # Y-axis limits with padding (start from 0)
    ymax = max(mean.max(), baseline_mean) * 1.1
    ax.set_ylim(0, ymax)

# Common legend outside subplots
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor=blue_shades[3], edgecolor='white', label='Coarse-grained'),
    Line2D([0], [0], color=baseline_color, linestyle='--', linewidth=1.5, label='1000-class')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, frameon=False, 
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "encoding_score_by_classes.png"
plt.savefig(output_path, dpi=300)
plt.savefig(output_path.with_suffix(".pdf"))
print(f"Saved to {output_path}")
plt.show()
