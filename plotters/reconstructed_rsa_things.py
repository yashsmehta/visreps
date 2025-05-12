import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

# --- Load Data ---
df_combined = pd.read_csv("logs/pc_reconstruction_analysis.csv")

# --- Filter for THINGS dataset (early filter) ---
# Assuming 'neural_dataset' column exists. If not, this might need adjustment
# or an assumption that the CSV is already THINGS-specific.
if 'neural_dataset' in df_combined.columns:
    df_combined = df_combined[df_combined['neural_dataset'].astype(str).str.strip().str.lower() == 'things'].copy()
    if df_combined.empty:
        raise ValueError("No data found for 'things' dataset in pc_reconstruction_analysis.csv. Please check the CSV content and 'neural_dataset' column.")
else:
    print("Warning: 'neural_dataset' column not found in pc_reconstruction_analysis.csv. Assuming all data is for the THINGS dataset.")


# --- Split Data into df_cnn and df_pca based on 'pca_labels' column ---
if 'pca_labels' not in df_combined.columns:
    raise ValueError("'pca_labels' column not found in pc_reconstruction_analysis.csv. This column is needed to distinguish CNN and PCA model data.")

pca_labels_as_str_lower = df_combined['pca_labels'].astype(str).str.strip().str.lower()
df_cnn = df_combined[pca_labels_as_str_lower == 'false'].copy()
df_pca = df_combined[pca_labels_as_str_lower == 'true'].copy()

# --- Define Fixed Parameters for THINGS plotting ---
rsa_correlation_method = 'spearman' # Fixed as per request
layers_to_plot = ['fc1', 'fc2'] # MODIFIED: Only fc1 and fc2
pca_n_classes_values = sorted(df_pca['pca_n_classes'].unique().astype(int)) # Get from data, ensure sorted

# --- Setup Plot (1x2 grid) --- # MODIFIED
n_layers = len(layers_to_plot) # Will be 2
n_cols = 2 # MODIFIED
n_rows = 1 # MODIFIED

sns.set_theme(style="ticks", context="paper")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True, sharey=True) # MODIFIED figsize
axes_flat = axes.flatten() if n_layers > 1 else [axes] # Adjust for single plot or multiple

# Helper function to filter and get scores for a specific layer (to be refactored)
def get_plot_data_for_layer_things(df_cnn_layer_specific, df_pca_layer_specific, rsa_method):
    """
    Prepares data for plotting for a specific layer for the THINGS dataset.
    Assumes input dataframes are already filtered for the specific layer and 'things' dataset.
    Args:
        df_cnn_layer_specific: DataFrame for standard CNN model, filtered for the current layer.
        df_pca_layer_specific: DataFrame for PCA-trained models, filtered for the current layer.
        rsa_method: The RSA correlation method string (e.g., 'spearman').
    Returns:
        A dictionary containing:
            'cnn_reconstructed': DataFrame with 'pca_k' and 'score' for reconstructed standard CNN.
            'cnn_full_rsm_score': Single float score for full RSM of standard CNN.
            'pca_reconstructed': Dict mapping pca_n_classes to DataFrames 
                                 (with 'pca_k', 'score') for reconstructed PCA models.
            'pca_full_rsm': Dict mapping pca_n_classes to float scores for full RSM PCA models.
    """
    
    plot_data = {
        'cnn_reconstructed': pd.DataFrame(columns=['pca_k', 'score']),
        'cnn_full_rsm_score': None,
        'pca_reconstructed': {},
        'pca_full_rsm': {}
    }

    # --- Process Standard CNN Data (df_cnn_layer_specific) ---
    # Filter by RSA method
    cnn_filtered = df_cnn_layer_specific[
        df_cnn_layer_specific['compare_rsm_correlation'].astype(str).str.strip().str.lower() == rsa_method.lower()
    ].copy()

    # Reconstructed scores for CNN
    cnn_recon = cnn_filtered[cnn_filtered['reconstruct_from_pcs'] == True]
    if not cnn_recon.empty:
        # Assuming 'score' is the direct RSA score, and 'pca_k' is present.
        # If there are multiple entries for the same pca_k (e.g. different subjects, though we are ignoring subject now),
        # we should average them. For THINGS, this might not be an issue if data is already averaged or single-entry.
        # For now, let's assume data is ready or take mean if multiple scores per pca_k.
        plot_data['cnn_reconstructed'] = cnn_recon.groupby('pca_k')['score'].mean().reset_index().sort_values('pca_k')

    # Full RSM score for CNN
    cnn_full = cnn_filtered[cnn_filtered['reconstruct_from_pcs'] == False]
    if not cnn_full.empty:
        # Again, average if multiple scores exist (e.g. if 'pca_k' column is present but irrelevant for full RSM)
        plot_data['cnn_full_rsm_score'] = cnn_full['score'].mean()

    # --- Process PCA-trained Models Data (df_pca_layer_specific) ---
    # Filter by RSA method
    pca_filtered = df_pca_layer_specific[
        df_pca_layer_specific['compare_rsm_correlation'].astype(str).str.strip().str.lower() == rsa_method.lower()
    ].copy()

    for n_classes in pca_n_classes_values: # Use the globally defined unique values
        df_n_class_specific = pca_filtered[pca_filtered['pca_n_classes'] == n_classes]
        if df_n_class_specific.empty:
            continue

        # Reconstructed scores for this PCA model
        pca_recon_n_class = df_n_class_specific[df_n_class_specific['reconstruct_from_pcs'] == True]
        if not pca_recon_n_class.empty:
            plot_data['pca_reconstructed'][n_classes] = pca_recon_n_class.groupby('pca_k')['score'].mean().reset_index().sort_values('pca_k')
        
        # Full RSM score for this PCA model
        pca_full_n_class = df_n_class_specific[df_n_class_specific['reconstruct_from_pcs'] == False]
        if not pca_full_n_class.empty:
            plot_data['pca_full_rsm'][n_classes] = pca_full_n_class['score'].mean()
            
    return plot_data

# --- Define Colors ---
# Color for standard CNN (back to red)
cnn_color = 'red' 
# pca_avg_recon_color = 'red' # REMOVED

# Colors for PCA n-classes lines (shades of blue)
if pca_n_classes_values:
    blue_shades = plt.cm.Blues(np.linspace(0.3, 0.9, len(pca_n_classes_values)))
    pca_colors = {n_class: blue_shades[i] for i, n_class in enumerate(pca_n_classes_values)}
else:
    pca_colors = {}

legend_handles_main_lines = []
legend_handles_full_rsm = []

# --- Loop through layers and plot ---
for i, layer_name in enumerate(layers_to_plot):
    ax = axes_flat[i]
    
    # Filter data for the current layer
    df_cnn_layer = df_cnn[df_cnn['layer'] == layer_name].copy()
    df_pca_layer = df_pca[df_pca['layer'] == layer_name].copy()

    if df_cnn_layer.empty and df_pca_layer.empty:
        print(f"No data found for layer {layer_name}. Skipping subplot.")
        ax.set_title(f"{layer_name.upper()} (No Data)", fontsize=10)
        ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        continue

    layer_plot_data = get_plot_data_for_layer_things(df_cnn_layer, df_pca_layer, rsa_correlation_method)

    # Plot Standard CNN (1k-way) Reconstructed (now cnn_color)
    cnn_recon_data = layer_plot_data['cnn_reconstructed']
    cnn_recon_label = '1k-way'
    if not cnn_recon_data.empty:
        ax.plot(cnn_recon_data['pca_k'], cnn_recon_data['score'], marker='o', markersize=5, 
                linewidth=2, label=cnn_recon_label, color=cnn_color, zorder=3)
        if i == 0: 
             legend_handles_main_lines.append(mlines.Line2D([], [], color=cnn_color, marker='o', linestyle='-', label=cnn_recon_label))

    # Plot Standard CNN (1k-way) Full RSM (now cnn_color)
    cnn_full_score = layer_plot_data['cnn_full_rsm_score']
    cnn_full_label = '1k-way (Full)'
    if cnn_full_score is not None:
        ax.axhline(cnn_full_score, color=cnn_color, linestyle='--', linewidth=2, label=cnn_full_label, zorder=2.5)
        if i == 0: 
            legend_handles_full_rsm.append(mlines.Line2D([], [], color=cnn_color, linestyle='--', label=cnn_full_label))

    # Plot PCA-derived Models Reconstructed (Individual n_classes)
    for n_classes, recon_data in layer_plot_data['pca_reconstructed'].items():
        if not recon_data.empty:
            color = pca_colors.get(n_classes, 'gray')
            label = f'PCA {n_classes}-way'
            ax.plot(recon_data['pca_k'], recon_data['score'], marker='s', markersize=4, 
                    linewidth=1.5, label=label, color=color, zorder=2)
            if i == 0: 
                 legend_handles_main_lines.append(mlines.Line2D([], [], color=color, marker='s', linestyle='-', label=label))

    # Plot PCA-derived Models Full RSM (Individual n_classes)
    for n_classes, full_score in layer_plot_data['pca_full_rsm'].items():
        if full_score is not None:
            color = pca_colors.get(n_classes, 'gray') 
            label = f'PCA {n_classes}-way (Full)'
            ax.axhline(full_score, color=color, linestyle=':', linewidth=1.5, label=label, zorder=1.5)
            if i == 0: 
                legend_handles_full_rsm.append(mlines.Line2D([], [], color=color, linestyle=':', label=label))

    # --- Customize Subplot Appearance ---
    ax.set_title(layer_name.upper(), fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Set x-axis major and minor ticks
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Y-label only for the first plot in the row (leftmost)
    if i == 0:
        ax.set_ylabel('RSA Score (Spearman)', fontsize=12)
    # X-label for all plots in this configuration
    ax.set_xlabel('Reconstruction PCs (k)', fontsize=12)

# --- Clean up empty subplots if any --- # This section might not be needed for a fixed 1x2 grid
# for j in range(n_layers, n_rows * n_cols):
#    if n_rows * n_cols > n_layers : # only if there are actually empty subplots
#        fig.delaxes(axes_flat[j])

# --- Final Figure Customization ---
# Create a consolidated legend
# Sort handles to ensure consistent order if desired
# Example sort order: CNN Recon, CNN Full, then PCA Recon sorted, then PCA Full sorted
all_legend_handles = []

# Order for main lines: 1k-way recon, PCA n_class recon sorted
def sort_key_main_lines(handle):
    label = handle.get_label()
    if label.startswith('1k-way'): return (0, label)
    # if label.startswith('PCA-Models'): return (1, label) # REMOVED
    if label.startswith('PCA'): 
        try: return (1, int(label.split(' ')[1].split('-')[0]), label) # Index changed from 2 to 1
        except: return (1, float('inf'), label) # Index changed from 2 to 1, Fallback for unexpected format
    return (2, label) # Index changed from 3 to 2, Fallback

legend_handles_main_lines.sort(key=sort_key_main_lines)
all_legend_handles.extend(legend_handles_main_lines)

# Order for full RSM lines: 1k-way full, then PCA n_class full sorted
def sort_key_full_lines(handle):
    label = handle.get_label()
    if label.startswith('1k-way'): return (0, label)
    if label.startswith('PCA'): 
        try: return (1, int(label.split(' ')[1].split('-')[0]), label)
        except: return (1, float('inf'), label) # Fallback for unexpected format
    return (2, label) # Fallback

legend_handles_full_rsm.sort(key=sort_key_full_lines)
all_legend_handles.extend(legend_handles_full_rsm)


if all_legend_handles:
    # Remove duplicate labels if any, keeping the first instance
    unique_handles_labels = {}
    for handle in all_legend_handles:
        if handle.get_label() not in unique_handles_labels:
            unique_handles_labels[handle.get_label()] = handle
    
    fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), 
               loc='lower center', bbox_to_anchor=(0.5, 0.03),
               ncol=max(1, len(unique_handles_labels)//2), 
               fontsize=10, title="Model Type", title_fontsize=12)

fig.suptitle(f'RSA Score vs. Reconstruction PCs for THINGS Dataset (FC Layers)', fontsize=18, weight='bold')
fig.tight_layout(rect=[0, 0.15, 1, 0.92])

# --- Save Figure ---
save_filename = 'plotters/rsa_reconstruction_vs_full_THINGS_fc_layers.png'
plt.savefig(save_filename, dpi=300)
print(f"Plot saved to {save_filename}")
# plt.show() # Comment out for non-interactive runs
