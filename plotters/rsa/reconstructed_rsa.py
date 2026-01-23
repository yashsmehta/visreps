"""RSA Reconstruction Analysis

Unified script for analyzing RSA scores with PC reconstruction.
Supports both NSD and THINGS datasets via configuration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

# ============================================================================
# CONFIGURATION - Change these to switch between NSD and THINGS analysis
# ============================================================================
DATASET = 'nsd'  # Options: 'nsd', 'things'

# Dataset-specific settings
DATASET_CONFIG = {
    'nsd': {
        'roi': 'early visual stream',
        'num_subjects': 8,
        'layers_to_plot': ['conv4'],  # Single layer for NSD
        'output_suffix': 'nsd',
        'figsize': (8, 6),
        'n_cols': 1,
    },
    'things': {
        'roi': None,  # THINGS doesn't use ROI
        'num_subjects': None,  # Not subject-averaged
        'layers_to_plot': ['fc1', 'fc2'],  # FC layers for THINGS
        'output_suffix': 'THINGS_fc_layers',
        'figsize': (12, 6),
        'n_cols': 2,
    }
}

# ============================================================================
# Load and prepare data
# ============================================================================
df_combined = pd.read_csv("logs/pc_reconstruction_analysis.csv")

# Filter for specified dataset if needed
if DATASET == 'things' and 'neural_dataset' in df_combined.columns:
    df_combined = df_combined[df_combined['neural_dataset'].astype(str).str.strip().str.lower() == 'things'].copy()
    if df_combined.empty:
        raise ValueError("No data found for 'things' dataset in pc_reconstruction_analysis.csv.")

# Split data into CNN and PCA based on 'pca_labels' column
if 'pca_labels' not in df_combined.columns:
    raise ValueError("'pca_labels' column not found. This column is needed to distinguish CNN and PCA model data.")

pca_labels_as_str_lower = df_combined['pca_labels'].astype(str).str.strip().str.lower()
df_cnn = df_combined[pca_labels_as_str_lower == 'false'].copy()
df_pca = df_combined[pca_labels_as_str_lower == 'true'].copy()

# Get configuration
config = DATASET_CONFIG[DATASET]
rsa_correlation_method = 'spearman'
layers_to_plot = config['layers_to_plot']
pca_n_classes_values = sorted(df_pca['pca_n_classes'].unique().astype(int)) if not df_pca.empty else []

# ============================================================================
# Helper function for NSD (subject-averaged)
# ============================================================================
def get_layer_scores_nsd(df_cnn, df_pca, layer, roi, rsa_method, num_subjects):
    """Get scores for NSD dataset, averaged across subjects."""
    all_cnn_recon_dfs_subj = []
    all_cnn_full_dfs_subj = []
    all_pca_recon_dfs_subj = []
    all_pca_full_dfs_subj = []

    for subj_idx in range(num_subjects):
        # CNN Reconstructed
        cnn_recon_cond = (
            (df_cnn['layer'] == layer) &
            (df_cnn['neural_dataset'] == 'nsd') &
            (df_cnn['region'] == roi) &
            (df_cnn['subject_idx'] == subj_idx) &
            (df_cnn['compare_rsm_correlation'].str.lower() == rsa_method.lower()) &
            (df_cnn['reconstruct_from_pcs'] == True)
        )
        cnn_recon_df = df_cnn[cnn_recon_cond]
        if not cnn_recon_df.empty:
            all_cnn_recon_dfs_subj.append(cnn_recon_df)

        # CNN Full
        cnn_full_cond = (
            (df_cnn['layer'] == layer) &
            (df_cnn['neural_dataset'] == 'nsd') &
            (df_cnn['region'] == roi) &
            (df_cnn['subject_idx'] == subj_idx) &
            (df_cnn['compare_rsm_correlation'].str.lower() == rsa_method.lower()) &
            (df_cnn['reconstruct_from_pcs'] == False)
        )
        cnn_full_df = df_cnn[cnn_full_cond]
        if not cnn_full_df.empty:
            all_cnn_full_dfs_subj.append(cnn_full_df)

        # PCA Reconstructed
        pca_recon_cond = (
            (df_pca['layer'] == layer) &
            (df_pca['neural_dataset'] == 'nsd') &
            (df_pca['region'] == roi) &
            (df_pca['subject_idx'] == subj_idx) &
            (df_pca['compare_rsm_correlation'].str.lower() == rsa_method.lower()) &
            (df_pca['reconstruct_from_pcs'] == True)
        )
        pca_recon_df = df_pca[pca_recon_cond]
        if not pca_recon_df.empty:
            all_pca_recon_dfs_subj.append(pca_recon_df)

        # PCA Full
        pca_full_cond = (
            (df_pca['layer'] == layer) &
            (df_pca['neural_dataset'] == 'nsd') &
            (df_pca['region'] == roi) &
            (df_pca['subject_idx'] == subj_idx) &
            (df_pca['compare_rsm_correlation'].str.lower() == rsa_method.lower()) &
            (df_pca['reconstruct_from_pcs'] == False)
        )
        pca_full_df = df_pca[pca_full_cond]
        if not pca_full_df.empty:
            all_pca_full_dfs_subj.append(pca_full_df)

    # Aggregate CNN Reconstructed
    if all_cnn_recon_dfs_subj:
        combined = pd.concat(all_cnn_recon_dfs_subj)
        cnn_recon_avg = combined.groupby('pca_k')['score'].mean().reset_index().sort_values('pca_k')
    else:
        cnn_recon_avg = pd.DataFrame(columns=['pca_k', 'score'])

    # Aggregate CNN Full
    cnn_full_avg = None
    if all_cnn_full_dfs_subj:
        combined = pd.concat(all_cnn_full_dfs_subj)
        cnn_full_avg = combined['score'].mean()

    # Aggregate PCA Full
    pca_full_avg = {}
    if all_pca_full_dfs_subj:
        combined = pd.concat(all_pca_full_dfs_subj)
        pca_full_avg = combined.groupby('pca_n_classes')['score'].mean().to_dict()

    return cnn_recon_avg, cnn_full_avg, pca_full_avg

# ============================================================================
# Helper function for THINGS (not subject-averaged)
# ============================================================================
def get_layer_scores_things(df_cnn_layer, df_pca_layer, rsa_method):
    """Get scores for THINGS dataset."""
    result = {
        'cnn_reconstructed': pd.DataFrame(columns=['pca_k', 'score']),
        'cnn_full_rsm_score': None,
        'pca_reconstructed': {},
        'pca_full_rsm': {}
    }

    # Filter by RSA method
    cnn_filtered = df_cnn_layer[
        df_cnn_layer['compare_rsm_correlation'].astype(str).str.strip().str.lower() == rsa_method.lower()
    ].copy()

    # CNN Reconstructed
    cnn_recon = cnn_filtered[cnn_filtered['reconstruct_from_pcs'] == True]
    if not cnn_recon.empty:
        result['cnn_reconstructed'] = cnn_recon.groupby('pca_k')['score'].mean().reset_index().sort_values('pca_k')

    # CNN Full
    cnn_full = cnn_filtered[cnn_filtered['reconstruct_from_pcs'] == False]
    if not cnn_full.empty:
        result['cnn_full_rsm_score'] = cnn_full['score'].mean()

    # PCA models
    pca_filtered = df_pca_layer[
        df_pca_layer['compare_rsm_correlation'].astype(str).str.strip().str.lower() == rsa_method.lower()
    ].copy()

    for n_classes in pca_n_classes_values:
        df_n_class = pca_filtered[pca_filtered['pca_n_classes'] == n_classes]
        if df_n_class.empty:
            continue

        # Reconstructed
        pca_recon = df_n_class[df_n_class['reconstruct_from_pcs'] == True]
        if not pca_recon.empty:
            result['pca_reconstructed'][n_classes] = pca_recon.groupby('pca_k')['score'].mean().reset_index().sort_values('pca_k')

        # Full
        pca_full = df_n_class[df_n_class['reconstruct_from_pcs'] == False]
        if not pca_full.empty:
            result['pca_full_rsm'][n_classes] = pca_full['score'].mean()

    return result

# ============================================================================
# Setup plot
# ============================================================================
sns.set_theme(style="ticks", context="paper")
n_layers = len(layers_to_plot)
n_cols = config['n_cols']
n_rows = math.ceil(n_layers / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=config['figsize'], sharex=True, sharey=True)
if n_layers == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_layers > 1 else [axes]

# Colors
cnn_color = 'red'
target_pca_n_classes = [2, 4, 8, 16, 32, 64]
if target_pca_n_classes:
    blue_shades = plt.cm.Blues(np.linspace(0.3, 0.9, len(target_pca_n_classes)))
    pca_colors = {n: blue_shades[i] for i, n in enumerate(target_pca_n_classes)}
else:
    pca_colors = {}

legend_handles = []

# ============================================================================
# Plot for each layer
# ============================================================================
for i, layer_name in enumerate(layers_to_plot):
    ax = axes[i]
    print(f"--- Processing Layer: {layer_name} ---")

    if DATASET == 'nsd':
        cnn_recon_df, cnn_full_score, pca_full_scores = get_layer_scores_nsd(
            df_cnn, df_pca, layer_name, config['roi'], rsa_correlation_method, config['num_subjects']
        )

        min_x, max_x = 0.5, 6.5

        # Plot CNN Reconstructed
        if not cnn_recon_df.empty:
            cnn_recon_plot = cnn_recon_df[cnn_recon_df['pca_k'] <= 6].copy()
            if not cnn_recon_plot.empty:
                label = '1k-way (Reconstructed)'
                sns.lineplot(data=cnn_recon_plot, x='pca_k', y='score', marker='o', markersize=7,
                            linewidth=2.5, label=label, color=cnn_color, linestyle='-',
                            zorder=3, ax=ax, markeredgecolor='darkred')
                if i == 0:
                    legend_handles.append(mlines.Line2D([], [], color=cnn_color, marker='o', markersize=7,
                                                        linestyle='-', label=label, markeredgecolor='darkred'))

        # Plot CNN Full RSM
        if cnn_full_score is not None:
            label = '1k-way (Full RSM)'
            ax.plot([min_x, 6.1], [cnn_full_score, cnn_full_score],
                    color=cnn_color, linestyle='--', linewidth=2.5, zorder=2.5, label=label)
            if i == 0:
                legend_handles.append(mlines.Line2D([], [], color=cnn_color, linestyle='--', linewidth=2.5, label=label))

        # Plot PCA Full RSM lines
        for k_classes in target_pca_n_classes:
            score = pca_full_scores.get(k_classes)
            if score is not None:
                color = pca_colors.get(k_classes, 'gray')
                label = f'PCA {k_classes}-way (Full RSM)'
                ax.plot([min_x, 6.1], [score, score], color=color, linestyle='--', linewidth=2.0, zorder=2, label=label)
                if i == 0:
                    legend_handles.append(mlines.Line2D([], [], color=color, linestyle='--', linewidth=2.0, label=label))

        ax.set_xlim(min_x, max_x)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    else:  # THINGS
        df_cnn_layer = df_cnn[df_cnn['layer'] == layer_name].copy()
        df_pca_layer = df_pca[df_pca['layer'] == layer_name].copy()

        if df_cnn_layer.empty and df_pca_layer.empty:
            ax.set_title(f"{layer_name.upper()} (No Data)", fontsize=10)
            continue

        layer_data = get_layer_scores_things(df_cnn_layer, df_pca_layer, rsa_correlation_method)

        # CNN Reconstructed
        cnn_recon = layer_data['cnn_reconstructed']
        if not cnn_recon.empty:
            label = '1k-way'
            ax.plot(cnn_recon['pca_k'], cnn_recon['score'], marker='o', markersize=5,
                    linewidth=2, label=label, color=cnn_color, zorder=3)
            if i == 0:
                legend_handles.append(mlines.Line2D([], [], color=cnn_color, marker='o', linestyle='-', label=label))

        # CNN Full RSM
        cnn_full = layer_data['cnn_full_rsm_score']
        if cnn_full is not None:
            label = '1k-way (Full)'
            ax.axhline(cnn_full, color=cnn_color, linestyle='--', linewidth=2, label=label, zorder=2.5)
            if i == 0:
                legend_handles.append(mlines.Line2D([], [], color=cnn_color, linestyle='--', label=label))

        # PCA Reconstructed
        for n_classes, recon_data in layer_data['pca_reconstructed'].items():
            if not recon_data.empty:
                color = pca_colors.get(n_classes, 'gray')
                label = f'PCA {n_classes}-way'
                ax.plot(recon_data['pca_k'], recon_data['score'], marker='s', markersize=4,
                        linewidth=1.5, label=label, color=color, zorder=2)
                if i == 0:
                    legend_handles.append(mlines.Line2D([], [], color=color, marker='s', linestyle='-', label=label))

        # PCA Full RSM
        for n_classes, full_score in layer_data['pca_full_rsm'].items():
            if full_score is not None:
                color = pca_colors.get(n_classes, 'gray')
                label = f'PCA {n_classes}-way (Full)'
                ax.axhline(full_score, color=color, linestyle=':', linewidth=1.5, label=label, zorder=1.5)
                if i == 0:
                    legend_handles.append(mlines.Line2D([], [], color=color, linestyle=':', label=label))

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_title(layer_name.upper(), fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    if i == 0:
        ax.set_ylabel('RSA Score (Spearman)', fontsize=12)
    ax.set_xlabel('Reconstruction PCs (k)', fontsize=12)

sns.despine()

# ============================================================================
# Final figure customization
# ============================================================================
if DATASET == 'nsd':
    roi_str = config['roi'].replace(' ', '_').upper()
    fig.suptitle(f'RSA Analysis ({layers_to_plot[0].upper()}, NSD, {roi_str}, Avg Subj)',
                fontsize=16, weight='bold')
else:
    fig.suptitle(f'RSA Score vs. Reconstruction PCs for THINGS Dataset (FC Layers)',
                fontsize=18, weight='bold')

# Legend
if legend_handles:
    unique_handles = {h.get_label(): h for h in legend_handles}
    fig.legend(unique_handles.values(), unique_handles.keys(),
              loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=10, title='Model / RSM Type', title_fontsize='11')

fig.tight_layout(rect=[0, 0, 0.78, 0.93])

# Save
save_filename = f'plotters/plots/rsa_reconstruction_{config["output_suffix"]}.png'
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_filename}")
