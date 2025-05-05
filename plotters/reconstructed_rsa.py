import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

# --- Load Data ---
df_cnn = pd.read_csv("logs/eval/checkpoint/imagenet_cnn.csv")
df_pca = pd.read_csv("logs/eval/checkpoint/imagenet_pca.csv")

# --- Define Fixed Filters ---
neural_dataset = 'nsd'
roi = 'early visual stream'
subject_idx = 0
rsa_correlation_method = 'spearman'

# Define layers to plot
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']
n_layers = len(layers)
n_cols = 4 # Use 4 columns for better layout (FC1, FC2 side-by-side)
n_rows = math.ceil(n_layers / n_cols)

# Helper function to filter and get scores for a specific layer
def get_layer_scores(df_cnn, df_pca, layer, roi, subject_idx, neural_dataset, rsa_correlation_method):
    """Filters dataframes for a specific layer and returns baseline and reconstructed scores."""

    # --- Filter CNN Data (Baseline) ---
    cnn_filter_conditions = (
        (df_cnn['layer'] == layer) &
        (df_cnn['neural_dataset'] == neural_dataset) &
        (df_cnn['region'] == roi) &
        (df_cnn['subject_idx'] == subject_idx) &
        (df_cnn['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
        (df_cnn['reconstruct_from_pcs'] == False)
    )
    cnn_base_row = df_cnn[cnn_filter_conditions]
    cnn_base_score = cnn_base_row['score'].iloc[0] if not cnn_base_row.empty else None

    # --- Filter PCA Data (Reconstructed) ---
    pca_filter_conditions = (
        (df_pca['layer'] == layer) &
        (df_pca['neural_dataset'] == neural_dataset) &
        (df_pca['region'] == roi) &
        (df_pca['subject_idx'] == subject_idx) &
        (df_pca['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
        (df_pca['reconstruct_from_pcs'] == True)
    )
    pca_recon_df = df_pca[pca_filter_conditions].sort_values(['pca_n_classes', 'pca_k'])

    # Group reconstructed data by pca_n_classes
    pca_recon_grouped = {
        k_classes: group for k_classes, group in pca_recon_df.groupby('pca_n_classes')
    }

    return cnn_base_score, pca_recon_grouped

# --- Setup Plot ---
sns.set_theme(style="ticks", context="paper")
# Create subplots without sharing y-axis
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
axes = axes.flatten() # Flatten the axes array for easy iteration

# Define colors for PCA k-ways - get unique values across all layers first
all_pca_n_classes = sorted(df_pca[
    (df_pca['neural_dataset'] == neural_dataset) &
    (df_pca['region'] == roi) &
    (df_pca['subject_idx'] == subject_idx) &
    (df_pca['reconstruct_from_pcs'] == True) &
    (df_pca['layer'].isin(layers))
]['pca_n_classes'].unique())

# Use sequential blue colormap as requested
if all_pca_n_classes:
    n_pca_variants = len(all_pca_n_classes)
    # Adjust linspace for better visual separation if many variants
    blues_cmap = plt.cm.Blues(np.linspace(0.3, 1.0, n_pca_variants))
    pca_color_map = {k: color for k, color in zip(all_pca_n_classes, blues_cmap)}
else:
    pca_color_map = {}

baseline_color = 'black' # Color for the baseline CNN (1000-way)

legend_handles = []
has_added_handles = False # Flag to add handles only once

# --- Loop through layers and plot ---
for i, layer in enumerate(layers):
    ax = axes[i]
    print(f"--- Processing Layer: {layer} ---")

    # --- Get Data for this layer ---
    cnn_base_score, pca_recon_grouped = get_layer_scores(
        df_cnn, df_pca, layer, roi, subject_idx, neural_dataset, rsa_correlation_method
    )

    print(f"Baseline Score (1000-way): {cnn_base_score if cnn_base_score is not None else 'N/A'}")

    all_x_ticks_layer = set() # Collect unique x-ticks for this layer
    min_x_layer, max_x_layer = float('inf'), float('-inf') # Track min/max x for limits

    # Plot reconstructed data (solid lines, markers for each pca_n_classes)
    for k_classes, recon_df in pca_recon_grouped.items():
        if recon_df.empty:
            print(f"Warning: No reconstructed data found for {k_classes}-way in layer {layer}.")
            continue

        print(f"Reconstructed Data ({k_classes}-way):")
        print(recon_df[['pca_k', 'score']].to_string())

        color = pca_color_map.get(k_classes, 'gray') # Get color or default
        label = f'{k_classes}-way'
        sns.lineplot(
            data=recon_df,
            x='pca_k',
            y='score',
            marker='o',
            markersize=5,
            linewidth=2.0,
            label=label, # Label for legend handle generation
            color=color,
            linestyle='-',
            zorder=2,
            ax=ax
        )
        all_x_ticks_layer.update(recon_df['pca_k'].unique())
        # Update min/max x values
        if not recon_df.empty:
            min_x_layer = min(min_x_layer, recon_df['pca_k'].min())
            max_x_layer = max(max_x_layer, recon_df['pca_k'].max())

        # Add legend handles only once
        if not has_added_handles:
            legend_handles.append(mlines.Line2D([], [], color=color, marker='o', markersize=5, linestyle='-', label=label))

    # Plot baseline CNN data (dashed line)
    if cnn_base_score is not None:
        # Determine x-range based on plotted data for this layer
        if all_x_ticks_layer:
             # xmin = min(all_x_ticks_layer)
             # xmax = max(all_x_ticks_layer)
             # Use tracked min/max for baseline line range
             xmin_line = min(all_x_ticks_layer)
             xmax_line = max(all_x_ticks_layer)
             # Also update overall min/max for axis limits
             min_x_layer = min(min_x_layer, xmin_line)
             max_x_layer = max(max_x_layer, xmax_line)

             ax.hlines(
                 cnn_base_score, xmin=xmin_line, xmax=xmax_line, # Use hlines for clarity
                 linestyle='--',
                 color=baseline_color,
                 linewidth=2.0,
                 alpha=0.9,
                 zorder=1,
                 label='_nolegend_' # Avoid duplicate legend entry
             )
        else: # Plot as axhline if no recon data was plotted
             ax.axhline(
                 cnn_base_score,
                 linestyle='--',
                 color=baseline_color,
                 linewidth=2.0,
                 alpha=0.9,
                 zorder=1,
                 label='_nolegend_'
             )


        # Add baseline legend handle only once
        if not has_added_handles:
            legend_handles.append(mlines.Line2D([], [], color=baseline_color, linestyle='--', linewidth=2.0, label='1000-way (Baseline)'))
    else:
        print(f"Warning: No baseline score found for layer {layer}.")

    has_added_handles = True # Ensure handles are added only in the first iteration

    # --- Customize Subplot Appearance ---
    ax.set_title(f'{layer.upper()}', fontsize=12, weight='normal')
    ax.set_xlabel('Number of Principal Components (k)', fontsize=11)
    ax.set_ylabel('RSA Score', fontsize=11)

    # Set x-ticks based on collected values for the layer
    # if all_x_ticks_layer:
    #     unique_ticks = np.sort(list(all_x_ticks_layer))
    #     # Ensure ticks are integers if they are whole numbers
    #     int_ticks = unique_ticks[unique_ticks == unique_ticks.astype(int)]
    #     ax.set_xticks(ticks=int_ticks.astype(int))
    #     ax.set_xticklabels(labels=int_ticks.astype(int), fontsize=9)
    # else:
    #     ax.set_xticks([])
    #     ax.set_xticklabels([])

    # Configure x-axis ticks (Major every 5, Minor every 1)
    if min_x_layer != float('inf') and max_x_layer != float('-inf'):
        # Set limits slightly padded
        pad = 1
        ax.set_xlim(min_x_layer - pad, max_x_layer + pad)
        # Set major and minor locators
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # Ensure only major ticks have labels
        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.tick_params(axis='x', which='minor', labelbottom=False)
    else:
        # No data, remove ticks
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    sns.despine(ax=ax)

    # --- Explicitly remove subplot legend ---
    if ax.get_legend():
        ax.get_legend().remove()

# Hide unused subplots if any
for j in range(n_layers, n_rows * n_cols):
    fig.delaxes(axes[j])

# --- Final Figure Customization ---
# Format ROI string for title/filename (replace space with underscore, capitalize)
roi_str_formatted = roi.replace(' ', '_').upper()
fig.suptitle(f'RSA Score vs Reconstruction PCs ({neural_dataset.upper()}, {roi_str_formatted}, Subj {subject_idx}, Corr Method: {rsa_correlation_method.capitalize()})', fontsize=16, weight='bold')

# Create and order the final common legend
if legend_handles:
    # Sort handles: PCA k-way first (numerically), then baseline
    pca_handles = sorted([h for h in legend_handles if h.get_linestyle() == '-'], key=lambda x: int(x.get_label().split('-')[0]))
    baseline_handle = [h for h in legend_handles if h.get_linestyle() == '--']
    ordered_handles = pca_handles + baseline_handle

# --- Create Legend BEFORE final layout adjustment ---
common_legend = None
if ordered_handles:
    common_legend = fig.legend(handles=ordered_handles, fontsize=10, title='Classification', title_fontsize='11', loc='center left', bbox_to_anchor=(0.9, 0.5)) # Anchor slightly left of edge initially
else:
    print("Warning: No legend handles generated, skipping figure legend.")

# --- Save Figure ---
# Apply tight_layout AFTER legend creation, constraining subplot area to leave space
fig.tight_layout(rect=[0, 0, 0.88, 0.95])
# Removed plt.subplots_adjust as tight_layout(rect=...) should handle it

# Format ROI string for filename (replace space with underscore, lowercase)
roi_str_filename = roi.replace(' ', '_').lower()
save_filename = f'plotters/reconstructed_rsa_layers_{roi_str_filename}_subj{subject_idx}_{neural_dataset}_corrmethod_{rsa_correlation_method}.png'
plt.savefig(save_filename, dpi=300)
print(f"Plot saved to {save_filename}")
# plt.show()
