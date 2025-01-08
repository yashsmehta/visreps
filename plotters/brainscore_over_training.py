import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Read data
df = pd.read_csv('logs/eval/checkpoint/base.csv')

# Filter data - we want RSA scores for different layers over epochs
df_filtered = df[
    (df['region'] == 'ventral visual stream') & 
    (df['analysis'] == 'rsa')
]

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create line plot with custom palette
palette = sns.color_palette("Set2", n_colors=len(df_filtered['layer'].unique()))
layer_colors = dict(zip(df_filtered['layer'].unique(), palette))

# Plot training lines
lineplot = sns.lineplot(
    data=df_filtered,
    x='epoch',
    y='score',
    hue='layer',
    palette=palette,
    linewidth=2.5,
    marker='o',
    ax=ax,
    legend=False  # We'll create custom legend
)

# Add reference stars
ax.plot(49, 0.2588, '*', color=layer_colors['conv5'], markersize=15, alpha=0.7)
ax.plot(49, 0.2981, '*', color=layer_colors['fc1'], markersize=15, alpha=0.7)

# Set x-axis limits
ax.set_xlim(0, 50)

# Customize plot
ax.set_title('AlexNet RSA Score over Training', pad=20, fontsize=16)
ax.set_xlabel('Training Epoch', fontsize=14)
ax.set_ylabel('RSA Score', fontsize=14)

# Create custom legend elements
legend_elements = []
# First add the line styles
legend_elements.extend([
    Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-', marker='o', label='Tiny-ImageNet'),
    Line2D([0], [0], color='gray', marker='*', markersize=10, linestyle='none', label='ImageNet')
])
# Then add the layer colors
legend_elements.extend([
    Line2D([0], [0], color=color, linewidth=2.5, label=layer)
    for layer, color in layer_colors.items()
])

# Add custom legend with two columns inside the plot
ax.legend(handles=legend_elements,
         ncol=2,
         title='Layers & Datasets',
         title_fontsize=11,
         fontsize=9,
         loc='lower right',
         frameon=True,
         borderpad=0.5,
         columnspacing=1.0,
         handlelength=1.5)

# Enhance grid and spines
ax.grid(True, linestyle='--', alpha=0.7)
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Adjust layout and save
plt.tight_layout()
plt.savefig('plotters/rsa_over_training.png', dpi=300, bbox_inches='tight')
plt.close()
