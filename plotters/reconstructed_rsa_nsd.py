import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

# --- Load Data ---
# df_cnn = pd.read_csv("logs/eval/checkpoint/imagenet_cnn.csv")
# df_pca = pd.read_csv("logs/eval/checkpoint/imagenet_pca.csv")
df_combined = pd.read_csv("logs/pc_reconstruction_analysis.csv")

# --- Split Data into df_cnn and df_pca based on 'pca_labels' column ---
# This logic is inspired by plotters/full-vs-pcs.py
if 'pca_labels' not in df_combined.columns:
    raise ValueError("'pca_labels' column not found in pc_reconstruction_analysis.csv. This column is needed to distinguish CNN and PCA model data.")

# Convert 'pca_labels' to string and lowercase for consistent filtering
# Assumes 'false' for standard CNN data (becomes df_cnn) and 'true' for PCA-derived model data (becomes df_pca)
pca_labels_as_str_lower = df_combined['pca_labels'].astype(str).str.strip().str.lower()

df_cnn = df_combined[pca_labels_as_str_lower == 'false'].copy()
df_pca = df_combined[pca_labels_as_str_lower == 'true'].copy()

# --- Define Fixed Filters ---
neural_dataset = 'nsd'
roi = 'early visual stream'
num_subjects_total = 8 # Define total number of subjects
rsa_correlation_method = 'spearman'

# Define layers to plot
layers = ['conv4', 'fc1']
n_layers = len(layers)
n_cols = 2 # Use 2 columns for side-by-side
n_rows = math.ceil(n_layers / n_cols)

# Helper function to filter and get scores for a specific layer
def get_layer_scores_avg(df_cnn, df_pca, layer, roi, neural_dataset, rsa_correlation_method, num_subjects):
    """Filters dataframes for a specific layer, averages scores across subjects,
    and returns:
    - cnn_recon_df_avg: DataFrame of reconstructed scores for standard CNN (pca_labels=False, recon=True)
    - cnn_full_score_avg: Single average score for full RSM of standard CNN (pca_labels=False, recon=False)
    - pca_recon_grouped_avg: Dict of DataFrames for reconstructed PCA model scores (pca_labels=True, recon=True)
    - pca_full_scores_avg: Dict of average scores for full RSM PCA models (pca_labels=True, recon=False)
    """

    all_cnn_recon_dfs_subj = []
    all_cnn_full_dfs_subj = [] # ADDED: For CNN full RSM
    all_pca_recon_dfs_subj = []
    all_pca_full_dfs_subj = []

    for subj_loop_idx in range(num_subjects):
        # --- Filter CNN Data (Reconstructed 1k-way, reconstruct_from_pcs == True) for current subject ---
        cnn_recon_filter_conditions_subj = (
            (df_cnn['layer'] == layer) &
            (df_cnn['neural_dataset'] == neural_dataset) &
            (df_cnn['region'] == roi) &
            (df_cnn['subject_idx'] == subj_loop_idx) &
            (df_cnn['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
            (df_cnn['reconstruct_from_pcs'] == True)
        )
        cnn_recon_df_for_subj = df_cnn[cnn_recon_filter_conditions_subj]
        if not cnn_recon_df_for_subj.empty:
            all_cnn_recon_dfs_subj.append(cnn_recon_df_for_subj)

        # --- ADDED: Filter CNN Data (Full 1k-way, reconstruct_from_pcs == False) for current subject ---
        cnn_full_filter_conditions_subj = (
            (df_cnn['layer'] == layer) &
            (df_cnn['neural_dataset'] == neural_dataset) &
            (df_cnn['region'] == roi) &
            (df_cnn['subject_idx'] == subj_loop_idx) &
            (df_cnn['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
            (df_cnn['reconstruct_from_pcs'] == False) # Key difference
        )
        cnn_full_df_for_subj = df_cnn[cnn_full_filter_conditions_subj]
        if not cnn_full_df_for_subj.empty:
            all_cnn_full_dfs_subj.append(cnn_full_df_for_subj)
        # --- END ADDED ---

        # --- Filter PCA Data (Reconstructed k-way, reconstruct_from_pcs == True) for current subject ---
        pca_recon_filter_conditions_subj = (
            (df_pca['layer'] == layer) &
            (df_pca['neural_dataset'] == neural_dataset) &
            (df_pca['region'] == roi) &
            (df_pca['subject_idx'] == subj_loop_idx) &
            (df_pca['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
            (df_pca['reconstruct_from_pcs'] == True)
        )
        pca_recon_df_for_subj = df_pca[pca_recon_filter_conditions_subj]
        if not pca_recon_df_for_subj.empty:
            all_pca_recon_dfs_subj.append(pca_recon_df_for_subj)
        
        # --- ADDED: Filter PCA Data (Full k-way, reconstruct_from_pcs == False) for current subject ---
        pca_full_filter_conditions_subj = (
            (df_pca['layer'] == layer) &
            (df_pca['neural_dataset'] == neural_dataset) &
            (df_pca['region'] == roi) &
            (df_pca['subject_idx'] == subj_loop_idx) &
            (df_pca['compare_rsm_correlation'].str.lower() == rsa_correlation_method.lower()) &
            (df_pca['reconstruct_from_pcs'] == False) # Key difference
        )
        pca_full_df_for_subj = df_pca[pca_full_filter_conditions_subj]
        if not pca_full_df_for_subj.empty:
            all_pca_full_dfs_subj.append(pca_full_df_for_subj)
        # --- END ADDED ---

    # --- Process CNN Data (Reconstructed - True, Average across subjects) ---
    if not all_cnn_recon_dfs_subj:
        cnn_recon_df_avg = pd.DataFrame(columns=['pca_k', 'score']) # Empty DF if no data
    else:
        combined_cnn_df = pd.concat(all_cnn_recon_dfs_subj)
        cnn_recon_df_avg = combined_cnn_df.groupby('pca_k')['score'].mean().reset_index()
        cnn_recon_df_avg = cnn_recon_df_avg.sort_values('pca_k')

    # --- ADDED: Process CNN Data (Full - False, Average across subjects) ---
    if not all_cnn_full_dfs_subj:
        cnn_full_score_avg = None # Single score, or None if no data
    else:
        combined_cnn_full_df = pd.concat(all_cnn_full_dfs_subj)
        # For "full" CNN data, score should ideally not depend on 'pca_k'.
        # It's a single score for the layer's full RSM.
        cnn_full_score_avg = combined_cnn_full_df['score'].mean()
    # --- END ADDED ---

    # --- Process PCA Data (Reconstructed - True, Average across subjects) ---
    if not all_pca_recon_dfs_subj:
        pca_recon_grouped_avg = {}
    else:
        combined_pca_recon_df = pd.concat(all_pca_recon_dfs_subj)
        pca_recon_df_avg = combined_pca_recon_df.groupby(['pca_n_classes', 'pca_k'])['score'].mean().reset_index()
        pca_recon_df_avg = pca_recon_df_avg.sort_values(['pca_n_classes', 'pca_k'])
        
        pca_recon_grouped_avg = {
            k_classes: group for k_classes, group in pca_recon_df_avg.groupby('pca_n_classes')
        }

    # --- ADDED: Process PCA Data (Full - False, Average across subjects) ---
    if not all_pca_full_dfs_subj:
        pca_full_scores_avg = {}
    else:
        combined_pca_full_df = pd.concat(all_pca_full_dfs_subj)
        # For "full" data, score should ideally not depend on 'pca_k' (reconstruction components).
        # We average scores per pca_n_classes.
        pca_full_df_avg_scores = combined_pca_full_df.groupby('pca_n_classes')['score'].mean()
        pca_full_scores_avg = pca_full_df_avg_scores.to_dict()
    # --- END ADDED ---

    return cnn_recon_df_avg, cnn_full_score_avg, pca_recon_grouped_avg, pca_full_scores_avg # MODIFIED return

# --- Setup Plot ---
sns.set_theme(style="ticks", context="paper")
# MODIFIED: Create a single subplot
fig, ax = plt.subplots(1, 1, figsize=(8, 6)) # Adjusted figsize for a single plot

# --- Define Fixed Layer for this plot ---
current_layer_to_plot = 'conv4'

# Define colors for PCA k-ways (now for full RSM dashed lines)
target_pca_n_classes_for_full_rsm = [2, 4, 8, 16, 32, 64] # MODIFIED: Added 64

# Check which of these target_pca_n_classes are actually available in the pca_full_scores data later
# Color map for PCA full RSM dashed lines
if target_pca_n_classes_for_full_rsm:
    n_pca_variants = len(target_pca_n_classes_for_full_rsm)
    blues_cmap = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants)) # Adjusted linspace for visibility
    pca_full_color_map = {k: color for k, color in zip(target_pca_n_classes_for_full_rsm, blues_cmap)}
else:
    pca_full_color_map = {}

baseline_color = 'red' # Color for the standard CNN (1k-way)

legend_handles = [] # Initialize list for legend handles

# --- Get Data for 'conv4' layer ---
print(f"--- Processing Layer: {current_layer_to_plot} ---")
cnn_recon_df, cnn_full_score, _, pca_full_layer_scores = get_layer_scores_avg(
    df_cnn, df_pca, current_layer_to_plot, roi, neural_dataset, rsa_correlation_method, num_subjects_total
) # pca_recon_grouped is not used for plotting anymore

min_x_plot_limit, max_x_plot_limit = 0.5, 6.5 # x-axis limits for 1-6 PCs

# --- Plot Standard CNN (1k-way) Data ---
print(f"--- Plotting: {current_layer_to_plot.upper()} (Standard 1k-way Labels) ---")
if not cnn_recon_df.empty:
    print(f"Reconstructed Data (Standard CNN) before filtering for x-axis:")
    print(cnn_recon_df[['pca_k', 'score']].to_string())
    
    # MODIFIED: Filter data for x-axis 1-6 PCs
    cnn_recon_df_plot = cnn_recon_df[cnn_recon_df['pca_k'] <= 6].copy()

    if not cnn_recon_df_plot.empty:
        cnn_label_recon = '1k-way (Reconstructed)'
        sns.lineplot(
            data=cnn_recon_df_plot, x='pca_k', y='score', marker='o', markersize=7, # Slightly larger marker
            linewidth=2.5, label=cnn_label_recon, color=baseline_color, linestyle='-', 
            zorder=3, ax=ax,
            markeredgecolor='darkred' # ADDED: Marker edge color
        )
        legend_handles.append(mlines.Line2D([], [], color=baseline_color, marker='o', markersize=7, 
                                            linestyle='-', label=cnn_label_recon, markeredgecolor='darkred'))
    else:
        print(f"Warning: No reconstructed data for Standard CNN in layer {current_layer_to_plot} for PCs 1-6.")
        
    # Original (unfiltered) data for reference if needed, but not plotted if beyond x_max_limit
    # all_x_ticks_data.update(cnn_recon_df['pca_k'].unique())
    # max_x_limit = max(max_x_limit, cnn_recon_df['pca_k'].max() + 0.5 if not cnn_recon_df['pca_k'].empty else max_x_limit)
else:
    print(f"Warning: No reconstructed data found for Standard CNN in layer {current_layer_to_plot}.")

if cnn_full_score is not None:
    cnn_label_full = '1k-way (Full RSM)' # Label for the dashed line
    ax.plot([min_x_plot_limit, 6.1], [cnn_full_score, cnn_full_score], 
            color=baseline_color, linestyle='--', linewidth=2.5, zorder=2.5, label=cnn_label_full)
    legend_handles.append(mlines.Line2D([], [], color=baseline_color, linestyle='--', linewidth=2.5, label=cnn_label_full))
    print(f"Full RSM Data (Standard CNN): score = {cnn_full_score:.6f}")
else:
    print(f"Warning: No full RSM data found for Standard CNN in layer {current_layer_to_plot}.")

# --- Plot PCA-derived Models (Full RSM scores as dashed horizontal lines) ---
print(f"--- Plotting: Full RSM Scores for PCA-derived Models (Layer {current_layer_to_plot.upper()}) ---")
for k_classes in target_pca_n_classes_for_full_rsm:
    full_score_val = pca_full_layer_scores.get(k_classes)
    color = pca_full_color_map.get(k_classes, 'gray')
    
    if full_score_val is not None:
        label = f'PCA {k_classes}-way (Full RSM)'
        ax.plot([min_x_plot_limit, 6.1], [full_score_val, full_score_val],
                color=color, linestyle='--', linewidth=2.0, zorder=2, label=label)
        legend_handles.append(mlines.Line2D([], [], color=color, linestyle='--', linewidth=2.0, label=label))
        print(f"Full RSM Data ({k_classes}-way, PCA Model): score = {full_score_val:.6f}")
    else:
        print(f"Info: No full RSM data found for PCA {k_classes}-way in layer {current_layer_to_plot}.")

# --- Customize Plot Appearance ---
ax.set_title(f'RSA Score vs. Reconstruction PCs ({current_layer_to_plot.upper()})', fontsize=14, weight='normal')
ax.set_xlabel('Number of Principal Components (k)', fontsize=12)
ax.set_ylabel('RSA Score', fontsize=12)

# Configure x-axis ticks (Major every 1, Minor every 0.5, focus 1-6)
ax.set_xlim(min_x_plot_limit, max_x_plot_limit) # MODIFIED: Use strict limits for 1-6 PCs
ax.xaxis.set_major_locator(MultipleLocator(1)) # Major ticks every 1 PC
# ax.xaxis.set_minor_locator(MultipleLocator(0.5)) # REMOVED: Minor ticks

ax.tick_params(axis='x', which='major', labelsize=10)
# ax.tick_params(axis='x', which='minor', labelbottom=False) # No minor ticks, so this is not needed
ax.tick_params(axis='y', labelsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))
ax.grid(True, linestyle=':', alpha=0.5) # MODIFIED: Lighter grid
sns.despine(ax=ax)

# --- Final Figure Customization ---
roi_str_formatted = roi.replace(' ', '_').upper()
fig.suptitle(f'RSA Analysis ({current_layer_to_plot.upper()}, {neural_dataset.upper()}, {roi_str_formatted}, Avg Subj)', fontsize=16, weight='bold')

# Create and order the final common legend
ordered_handles = []
if legend_handles:
    unique_legend_items = {handle.get_label(): handle for handle in legend_handles} # Deduplicate by label
    
    # Custom sort order: 1k-way recon, 1k-way full, then PCA fulls sorted by k
    handle_order_keys = ['1k-way (Reconstructed)', '1k-way (Full RSM)'] + \
                        [f'PCA {k}-way (Full RSM)' for k in sorted(target_pca_n_classes_for_full_rsm)]

    for key in handle_order_keys:
        if key in unique_legend_items:
            ordered_handles.append(unique_legend_items[key])

common_legend = None
if ordered_handles:
    common_legend = fig.legend(handles=ordered_handles, fontsize=10, title='Model / RSM Type', title_fontsize='11', loc='center left', bbox_to_anchor=(1.01, 0.5))
else:
    print("Warning: No legend handles generated, skipping figure legend.")

# --- Save Figure ---
# Adjust layout to make space for legend and suptitle
fig.tight_layout(rect=[0, 0, 0.78, 0.93]) # MODIFIED: Adjusted right margin further for legend with potentially fewer x-ticks space

roi_str_filename = roi.replace(' ', '_').lower()
save_filename = f'plotters/rsa_reconstruction_vs_full_{current_layer_to_plot}_{roi_str_filename}_avg_subj_{neural_dataset}.png'
plt.savefig(save_filename, dpi=300, bbox_inches='tight') # Added bbox_inches='tight' for safety
print(f"Plot saved to {save_filename}")
# plt.show()

# --- Loop through layers and plot --- # This entire loop is now replaced by the direct subplot handling above.
# for i, layer in enumerate(layers):
#    ax = axes[i]
#    print(f"--- Processing Layer: {layer} ---")

    # --- Get Data for this layer (averaged across subjects) ---
    # MODIFIED: Unpack new third return value
#    cnn_recon_df, pca_recon_grouped, pca_full_layer_scores = get_layer_scores_avg(
#        df_cnn, df_pca, layer, roi, neural_dataset, rsa_correlation_method, num_subjects_total
#    )
# ... (rest of the old loop is removed)
# ... existing code ...
