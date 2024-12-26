import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

# Read the CSV file
df = pd.read_csv('logs/eval/torchvision/base.csv')

# Set the style
sns.set_theme(style="whitegrid")

# Define color palette - different shade pairs for each region
region_palettes = {
    'early visual stream': ['#95a5a6', '#2c3e50'],      # Light and dark blue
    'midventral visual stream': ['#82e0aa', '#27ae60'],  # Light and dark green
    'ventral visual stream': ['#d2b4de', '#8e44ad']      # Light and dark purple
}

# Create the plot
g = sns.catplot(
    data=df,
    x='layer',
    y='score',
    hue='pretrained',
    col='region',
    kind='bar',
    height=4,
    aspect=1.2,
    palette=['#9b9b9b', '#4a4a4a'],  # Light then dark gray
    legend=False  # Remove default legend
)

# Customize the plot
g.set_xticklabels(rotation=45)
g.set_axis_labels('', 'RSA Score')  # Empty string for x since we use supxlabel

# Update the subplot titles to capitalize each word while keeping spaces
for ax, title in zip(g.axes.flat, df['region'].unique()):
    ax.set_title(title.title())

# For each subplot, update the color palette based on the region
for ax, region in zip(g.axes.flat, df['region'].unique()):
    bars = [patch for patch in ax.patches]
    n_groups = len(df['layer'].unique())
    for i in range(n_groups):
        if region in region_palettes:
            bars[i].set_facecolor(region_palettes[region][0])  # Untrained
            bars[i + n_groups].set_facecolor(region_palettes[region][1])  # Trained

# Create custom legend handles - one entry for each training status
untrained_patches = [Patch(facecolor=colors[0], label='Untrained') for colors in region_palettes.values()]
trained_patches = [Patch(facecolor=colors[1], label='Trained') for colors in region_palettes.values()]

# Add the custom legend with one label per training status
g.fig.legend([tuple(untrained_patches), tuple(trained_patches)], ['Untrained', 'Trained'],
           bbox_to_anchor=(1.02, 0.7),
           loc='center left',
           title='AlexNet Model',
           handler_map={tuple: HandlerTuple(ndivide=None)})

# Set common x-axis label with consistent size
g.fig.supxlabel('AlexNet Layer', size=12)  # Using a standard size that matches seaborn defaults

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plotters/imgs/layer_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as 'layer_analysis.png'")