import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# --- Plotting Aesthetics --- #
plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean base
plt.rcParams.update({
    'font.family': 'sans-serif', # Use a standard sans-serif font
    # 'font.sans-serif': ['Arial', 'Helvetica'], # Specify preferred fonts (Commented out)
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
rows, cols = np.triu_indices_from(data_rsm_1000, k=1)

data_upper_1000 = data_rsm_1000[rows, cols]
fc2_upper_1000 = fc2_rsm_1000[rows, cols]
data_upper_2 = data_rsm_2[rows, cols]
fc2_upper_2 = fc2_rsm_2[rows, cols]

print(f"Number of elements in upper triangle: {data_upper_1000.size}")

# --- Subsample 10% of the data --- #
n_elements = data_upper_1000.size
sample_size = int(0.1 * n_elements)
np.random.seed(42) # for reproducibility
indices = np.random.choice(n_elements, sample_size, replace=False)

data_upper_1000_sample = data_upper_1000[indices]
fc2_upper_1000_sample = fc2_upper_1000[indices]
data_upper_2_sample = data_upper_2[indices]
fc2_upper_2_sample = fc2_upper_2[indices]
rows_sample = rows[indices] # Keep corresponding row indices if needed for coloring (though density plot won't use it)

print(f"Subsampling to {sample_size} elements (10%)")


# --- Create Figure with Two Subplots --- #
fig, axes = plt.subplots(1, 2, figsize=(6, 3.375), sharex=True, sharey=True) # Reduced height (75% of 4.5)
fig.suptitle(f'{dataset.upper()} Data vs. FC2 RSM 2D Density', fontsize=12) # Adjusted fontsize for smaller fig

ax_density1 = axes[0]

# Plot 2D Density 1
kde1 = sns.kdeplot(
    x=data_upper_1000_sample, y=fc2_upper_1000_sample, ax=ax_density1,
    fill=True, cmap="viridis", thresh=0.05, # Adjust thresh as needed
    cbar=False # We'll add a common one later
)
ax_density1.set_xlabel('Recorded Data (Pearson)')
ax_density1.set_ylabel('FC2 Pearson Correlation')
ax_density1.grid(False)
# ax_scatter1.text(0.95, 0.05, '1000 Classes', transform=ax_scatter1.transAxes,
#                  fontsize=11, verticalalignment='bottom', horizontalalignment='right')
ax_density1.text(0.95, 0.05, '1000 Classes', transform=ax_density1.transAxes,
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right')

ax_density2 = axes[1]

# Plot 2D Density 2
kde2 = sns.kdeplot(
    x=data_upper_2_sample, y=fc2_upper_2_sample, ax=ax_density2,
    fill=True, cmap="viridis", thresh=0.05, # Adjust thresh as needed
    cbar=False # We'll add a common one later
)
ax_density2.set_xlabel('Recorded Data (Pearson)')
# ax_scatter2.tick_params(axis='y', labelleft=False) # Hide y-labels - Handled by sharey
ax_density2.grid(False)
# ax_scatter2.text(0.95, 0.05, '2 Classes', transform=ax_scatter2.transAxes,
#                  fontsize=11, verticalalignment='bottom', horizontalalignment='right')
ax_density2.text(0.95, 0.05, '2 Classes', transform=ax_density2.transAxes,
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right')


# --- Synchronize Axes and Add Identity Lines --- #
# Determine common limits based on the data ranges used in KDE plots
all_data = np.concatenate([
    data_upper_1000_sample, fc2_upper_1000_sample,
    data_upper_2_sample, fc2_upper_2_sample
])
min_val = np.min(all_data)
max_val = np.max(all_data)
lims = [min_val, 1.05] # Set upper limit to 1.05

for ax in axes:
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.0, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# Note: The colorbar represents the density scale implicitly shared by both plots
# due to identical cmap and thresh settings.
norm = plt.Normalize(vmin=0, vmax=1) # Placeholder norm, actual density scale is complex
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([]) # Necessary for colorbar creation

# Create a dedicated axes for the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(sm, cax=cbar_ax)

cbar.set_label('Density', size=10)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_linewidth(0.8)

# Adjust layout
fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Re-enable tight_layout, adjust top slightly for suptitle

# Save the plot
output_filename = f"plotters/plots/rsm_comparison_density_{dataset}_nature.png" # Updated filename
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Side-by-side 2D density plot saved to {output_filename}")
plt.close(fig) # Close the figure to free memory
