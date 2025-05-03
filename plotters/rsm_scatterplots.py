import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# --- Plotting Aesthetics --- #
plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean base
plt.rcParams.update({
    'font.family': 'sans-serif', # Use a standard sans-serif font
    'font.sans-serif': ['Arial', 'Helvetica'], # Specify preferred fonts
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.0, # Default line width
    'axes.linewidth': 0.8, # Axis spine thickness
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.major.pad': 4, # Padding between ticks and labels
    'ytick.major.pad': 4,
})

# --- Load Data --- #
dataset = "things"
data_1000cls = np.load(f"model_checkpoints/imagenet_cnn/cfg1/RSMs/rsms_{dataset}_epoch_10.npz")
data_2cls = np.load(f"model_checkpoints/imagenet_pca/cfg1/RSMs/rsms_{dataset}_epoch_10.npz")

# --- Extract RSMs --- #
# Assuming 'neural' and 'fc2' keys exist in both files
data_rsm_1000 = data_1000cls['neural']
fc2_rsm_1000 = data_1000cls['fc2']
data_rsm_2 = data_2cls['neural']
fc2_rsm_2 = data_2cls['fc2']

print("1000 Cls - Data RSM shape:", data_rsm_1000.shape)
print("1000 Cls - FC2 RSM shape:", fc2_rsm_1000.shape)
print("2 Cls - Data RSM shape:", data_rsm_2.shape)
print("2 Cls - FC2 RSM shape:", fc2_rsm_2.shape)

# --- Get Upper Triangle Elements --- #
# Assuming both RSMs have the same shape
rows, cols = np.triu_indices_from(data_rsm_1000, k=1)

data_upper_1000 = data_rsm_1000[rows, cols]
fc2_upper_1000 = fc2_rsm_1000[rows, cols]
data_upper_2 = data_rsm_2[rows, cols]
fc2_upper_2 = fc2_rsm_2[rows, cols]

print(f"Number of elements in upper triangle: {data_upper_1000.size}")

# --- Create Figure with Two Subplots --- #
fig = plt.figure(figsize=(12, 6)) # Wider figure for two plots

# Define GridSpec for layout (4 rows, 8 columns base)
gs = gridspec.GridSpec(4, 8, figure=fig, width_ratios=[3, 1, 0.1, 0.1, 3, 1, 0.1, 0.1], height_ratios=[1, 1, 1, 1]) # Adjusted width ratios for spacing

# --- Subplot 1: 1000 Classes --- #
ax_scatter1 = fig.add_subplot(gs[1:4, 0])
ax_kde_x1 = fig.add_subplot(gs[0, 0], sharex=ax_scatter1)
ax_kde_y1 = fig.add_subplot(gs[1:4, 1], sharey=ax_scatter1)

# Plot scatter 1
scatter1 = ax_scatter1.scatter(
    data_upper_1000, fc2_upper_1000, c=rows, cmap='viridis',
    alpha=0.15, s=3, edgecolors='none', rasterized=True
)
ax_scatter1.set_xlabel('Recorded Data Pearson Correlation')
ax_scatter1.set_ylabel('FC2 Pearson Correlation')
ax_scatter1.grid(False)
ax_scatter1.text(0.95, 0.05, '1000 Classes', transform=ax_scatter1.transAxes,
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right')

# Plot KDEs 1
sns.kdeplot(x=data_upper_1000, ax=ax_kde_x1, fill=True, color=sns.color_palette("viridis")[0], linewidth=1.0)
ax_kde_x1.tick_params(axis='x', labelbottom=False)
ax_kde_x1.set_ylabel('Density')
ax_kde_x1.set_title(f'{dataset.upper()} Data vs. FC2 RSM Scatterplots', fontsize=12, loc='center', x=1.1) # Centered title spanning plots


sns.kdeplot(y=fc2_upper_1000, ax=ax_kde_y1, fill=True, color=sns.color_palette("viridis")[2], linewidth=1.0)
ax_kde_y1.tick_params(axis='y', labelleft=False)
ax_kde_y1.set_xlabel('Density')

# --- Subplot 2: 2 Classes --- #
ax_scatter2 = fig.add_subplot(gs[1:4, 4], sharey=ax_scatter1) # Share y-axis with plot 1
ax_kde_x2 = fig.add_subplot(gs[0, 4], sharex=ax_scatter2)
ax_kde_y2 = fig.add_subplot(gs[1:4, 5], sharey=ax_scatter2)

# Plot scatter 2
scatter2 = ax_scatter2.scatter(
    data_upper_2, fc2_upper_2, c=rows, cmap='viridis',
    alpha=0.15, s=3, edgecolors='none', rasterized=True
)
ax_scatter2.set_xlabel('Recorded Data Pearson Correlation')
ax_scatter2.tick_params(axis='y', labelleft=False) # Hide y-labels
ax_scatter2.grid(False)
ax_scatter2.text(0.95, 0.05, '2 Classes', transform=ax_scatter2.transAxes,
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right')


# Plot KDEs 2
sns.kdeplot(x=data_upper_2, ax=ax_kde_x2, fill=True, color=sns.color_palette("viridis")[0], linewidth=1.0)
ax_kde_x2.tick_params(axis='x', labelbottom=False)
ax_kde_x2.set_ylabel('Density')
ax_kde_x2.tick_params(axis='y', labelleft=False) # Hide y-labels


sns.kdeplot(y=fc2_upper_2, ax=ax_kde_y2, fill=True, color=sns.color_palette("viridis")[2], linewidth=1.0)
ax_kde_y2.tick_params(axis='y', labelleft=False)
ax_kde_y2.set_xlabel('Density')

# --- Synchronize Axes and Add Identity Lines --- #
# Scatter plots
min_val = min(ax_scatter1.get_xlim()[0], ax_scatter1.get_ylim()[0], ax_scatter2.get_xlim()[0], ax_scatter2.get_ylim()[0])
max_val = max(ax_scatter1.get_xlim()[1], ax_scatter1.get_ylim()[1], ax_scatter2.get_xlim()[1], ax_scatter2.get_ylim()[1])
lims = [min_val, max_val]

for ax in [ax_scatter1, ax_scatter2]:
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.0, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# KDE plots
max_density_x = max(ax_kde_x1.get_ylim()[1], ax_kde_x2.get_ylim()[1])
max_density_y = max(ax_kde_y1.get_xlim()[1], ax_kde_y2.get_xlim()[1])

ax_kde_x1.set_ylim(0, max_density_x * 1.05)
ax_kde_x2.set_ylim(0, max_density_x * 1.05)
ax_kde_y1.set_xlim(0, max_density_y * 1.05)
ax_kde_y2.set_xlim(0, max_density_y * 1.05)


# --- Common Colorbar --- #
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # Adjusted position/size for common bar
cbar = fig.colorbar(scatter1, cax=cbar_ax) # Use scatter1 for color mapping reference
cbar.set_label('Stimuli ID (ordered by PC1)', size=10)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_linewidth(0.8)


# --- Final Aesthetics --- #
# Remove spines for KDE plots
sns.despine(ax=ax_kde_x1, left=True, bottom=True)
sns.despine(ax=ax_kde_y1, left=True, bottom=True)
sns.despine(ax=ax_kde_x2, left=True, bottom=True)
sns.despine(ax=ax_kde_y2, left=True, bottom=True)
ax_kde_x1.grid(False)
ax_kde_y1.grid(False)
ax_kde_x2.grid(False)
ax_kde_y2.grid(False)


# Adjust layout
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space for colorbar
plt.subplots_adjust(wspace=0.1, hspace=0.1) # Fine-tune spacing


# Save the plot
output_filename = "rsm_comparison_scatter_kde_nature.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Side-by-side scatter plot saved to {output_filename}")
plt.close(fig) # Close the figure to free memory
