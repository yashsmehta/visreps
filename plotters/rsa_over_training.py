import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set seaborn style for better aesthetics
sns.set_theme(style="ticks", context="paper", font_scale=1.2) # Adjusted font scale for subplot clarity
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.0 # Slightly thinner lines for subplots
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Define path for the single CSV file
combined_csv_path = 'logs/over_training_analysis.csv'

# Read the combined CSV file
try:
    df_all = pd.read_csv(combined_csv_path)
    print(f"Successfully loaded combined file: {combined_csv_path}")
except FileNotFoundError:
    print(f"Error: The file {combined_csv_path} was not found.")
    exit()
except Exception as e:
    print(f"Error reading CSV file {combined_csv_path}: {e}")
    exit()

# --- Debug: Print Columns and DTypes for the combined DataFrame ---
print("\n--- df_all Columns and DTypes ---")
if 'df_all' in locals(): # Check if df_all was loaded
    # print(df_all.info()) # Uncomment for detailed info if needed
    # Convert pca_n_classes to numeric. This is used for detailing PCA models.
    if 'pca_n_classes' in df_all.columns:
        df_all['pca_n_classes_numeric'] = pd.to_numeric(df_all['pca_n_classes'], errors='coerce')
    else:
        print("Warning: 'pca_n_classes' column not found. Details for PCA model variants might be missing or affect plotting.")
        df_all['pca_n_classes_numeric'] = np.nan # Add the column as nan so it exists

    # Split into df_full (standard models) and df_pca (PCA-based models) based on 'pca_labels'
    if 'pca_labels' not in df_all.columns:
        print("Error: 'pca_labels' column not found in the CSV. Cannot reliably split into standard and PCA models.")
        df_full = pd.DataFrame()
        df_pca = pd.DataFrame()
        # Consider exiting if this state is critical, though downstream code handles empty DFs
    else:
        # Assuming pca_labels contains boolean values or values pandas can interpret as boolean for comparison.
        # If 'pca_labels' could be strings 'True'/'False', a more robust conversion might be:
        # df_all['pca_labels_bool'] = df_all['pca_labels'].astype(str).str.lower().map({'true': True, 'false': False})
        # Then filter on df_all['pca_labels_bool']
        df_full = df_all[df_all['pca_labels'] == False].copy()
        df_pca = df_all[df_all['pca_labels'] == True].copy()

    print(f"  - df_full (standard models, pca_labels==False) rows: {len(df_full)}")
    print(f"  - df_pca (PCA models, pca_labels==True) rows: {len(df_pca)}")
else:
    # This case should ideally not be reached if exit() is called on read error
    print("Error: df_all not loaded. Cannot proceed with splitting.")
    df_full = pd.DataFrame() 
    df_pca = pd.DataFrame()
    # exit() # Consider exiting if this state is critical

print("-----------------------------------\n")
# --- End Debug and Splitting ---

# --- Configuration ---
roi = "ventral visual stream" # UPDATED
target_region = roi # Use consistent variable name
# subject_idx = 0 # Ensure this is treated as an integer # REMOVED for averaging
# target_subject = subject_idx # Use consistent variable name # REMOVED for averaging
layers_to_plot = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2'] # 7 layers for 2x4 grid
plot_epoch_min = 0
plot_epoch_max = 18
epoch_tick_step = 3
# --- End Configuration ---

# --- Define Additional Filters ---
neural_data = 'nsd'
# --- End Additional Filters ---

# --- Helper function for robust filtering ---
def filter_dataframe(df, roi_filter, neural_data_filter): # Removed subject_filter
    # Conversion of 'subject_idx' to numeric is no longer needed for filtering here.
    # If 'subject_idx' column has mixed types and causes issues in groupby later,
    # it might need cleaning, but for now, let's assume groupby handles it or it's clean.

    conditions = pd.Series(True, index=df.index)
    if 'region' in df.columns:
        conditions &= (df['region'].astype(str).str.strip() == roi_filter)
    else:
        print("Warning: 'region' column not found.")
        conditions &= False # Ensure no match if column missing

    # conditions &= (df['subject_idx_numeric'] == subject_filter) # REMOVED: No longer filtering by specific subject

    if 'neural_dataset' in df.columns:
        conditions &= (df['neural_dataset'].astype(str).str.strip().str.lower() == neural_data_filter.lower())
    else:
        print("Warning: 'neural_dataset' column not found.")
        conditions &= False # Ensure no match if column missing

    return df[conditions].copy()
# --- End Helper function ---

# Apply filtering separately
print("Applying filters (region and neural dataset only)...")
filtered_df_full = filter_dataframe(df_full, target_region, neural_data)
filtered_df_pca = filter_dataframe(df_pca, target_region, neural_data)

# --- Aggregate scores across subjects ---
grouping_keys_full = ['epoch', 'layer']
if not filtered_df_full.empty:
    if not all(col in filtered_df_full.columns for col in grouping_keys_full + ['score']):
        print("Warning: df_full missing essential columns for aggregation (epoch, layer, score). Skipping aggregation.")
        filtered_df_full = pd.DataFrame() # Ensure it's empty if columns missing
    else:
        if 'epoch' in filtered_df_full.columns:
            filtered_df_full['epoch'] = pd.to_numeric(filtered_df_full['epoch'], errors='coerce')
            filtered_df_full.dropna(subset=['epoch'], inplace=True)
            if not filtered_df_full.empty:
                 filtered_df_full['epoch'] = filtered_df_full['epoch'].astype(int)
        if not filtered_df_full.empty:
            filtered_df_full = filtered_df_full.groupby(grouping_keys_full, as_index=False)['score'].mean()
    print(f"Aggregated to {len(filtered_df_full)} rows in FULL data after averaging.")
else:
    print("FULL data is empty before aggregation.")

grouping_keys_pca = ['epoch', 'layer', 'pca_n_classes']
if not filtered_df_pca.empty:
    if not all(col in filtered_df_pca.columns for col in grouping_keys_pca + ['score']):
        print("Warning: df_pca missing essential columns for aggregation (epoch, layer, pca_n_classes, score). Skipping aggregation.")
        filtered_df_pca = pd.DataFrame() # Ensure it's empty if columns missing
    else:
        if 'epoch' in filtered_df_pca.columns:
            filtered_df_pca['epoch'] = pd.to_numeric(filtered_df_pca['epoch'], errors='coerce')
            filtered_df_pca.dropna(subset=['epoch'], inplace=True)
            if not filtered_df_pca.empty:
                filtered_df_pca['epoch'] = filtered_df_pca['epoch'].astype(int)
        if 'pca_n_classes' in filtered_df_pca.columns and not filtered_df_pca.empty :
            filtered_df_pca['pca_n_classes'] = pd.to_numeric(filtered_df_pca['pca_n_classes'], errors='coerce')
            filtered_df_pca.dropna(subset=['pca_n_classes'], inplace=True)
        
        if not filtered_df_pca.empty:
            filtered_df_pca = filtered_df_pca.groupby(grouping_keys_pca, as_index=False)['score'].mean()
    print(f"Aggregated to {len(filtered_df_pca)} rows in PCA data after averaging.")
else:
    print("PCA data is empty before aggregation.")
# --- End Aggregation ---


print(f"Found {len(filtered_df_full)} matching rows in FULL data post-aggregation.")
print(f"Found {len(filtered_df_pca)} matching rows in PCA data post-aggregation.")

# Check if we have data to plot
if filtered_df_full.empty and filtered_df_pca.empty:
    print(f"No data found matching filter conditions (or after aggregation) in either file:")
    print(f"  Region: {target_region}")
    # print(f"  Subject Index: {target_subject}") # REMOVED
    print(f"  Neural Dataset: {neural_data}")
    exit()

# Define color scheme
full_color = '#c0392b'  # Dark red for full model

# Create continuous color scale for PCA variants using blues
# Use filtered_df_pca to find unique classes
potential_pca_classes = []
if not filtered_df_pca.empty and 'pca_n_classes' in filtered_df_pca.columns:
    potential_pca_classes = sorted(filtered_df_pca['pca_n_classes'].dropna().unique())

if not potential_pca_classes:
    print(f"Warning: No valid pca_n_classes found in filtered PCA data for Region: {target_region} (averaged).")
    pca_colors = {}
else:
    # Convert potential floats/strings safely to int after ensuring they are valid
    valid_pca_classes = []
    for c in potential_pca_classes:
        try:
            valid_pca_classes.append(int(float(c))) # Handle potential floats represented as strings/objects
        except (ValueError, TypeError):
            print(f"Warning: Skipping non-numeric pca_n_classes value: {c}")
    valid_pca_classes = sorted(list(set(valid_pca_classes))) # Ensure uniqueness and sort

    if not valid_pca_classes:
        print(f"Warning: No valid *numeric* pca_n_classes found after conversion (averaged).")
        pca_colors = {}
    else:
        n_pca_variants = len(valid_pca_classes)
        blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_pca_variants))
        pca_colors = dict(zip(valid_pca_classes, blues))
        print(f"Found PCA classes in filtered data: {valid_pca_classes}")

# Create figure with 2x4 subplots
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
fig.suptitle(f"RSA Score vs. Epoch for {target_region.title()}, Averaged Across Subjects",
             fontsize=16, y=0.97, fontweight='medium')

legend_handles = []
legend_labels = []
handles_added = set()

# max_epoch = 0 # REMOVED: X-axis is fixed based on plot_epoch_max

# Check which columns are actually present for plotting
score_col = 'score' # Assume 'score' exists, add checks if needed
epoch_col = 'epoch' # Assume 'epoch' exists, add checks if needed
pca_n_classes_col = 'pca_n_classes' # Assume 'pca_n_classes' exists in df_pca
layer_col = 'layer' # Assume 'layer' exists

for i, layer in enumerate(layers_to_plot):
    row, col = divmod(i, 4)
    ax = axes[row, col]

    # Get data for this layer from both filtered dataframes
    layer_data_full = pd.DataFrame()
    if layer_col in filtered_df_full.columns:
        layer_data_full = filtered_df_full[filtered_df_full[layer_col] == layer].sort_values(epoch_col)

    layer_data_pca = pd.DataFrame()
    if layer_col in filtered_df_pca.columns:
        layer_data_pca = filtered_df_pca[filtered_df_pca[layer_col] == layer].sort_values(epoch_col)

    if layer_data_full.empty and layer_data_pca.empty:
        print(f"Warning: No data for layer '{layer}' in Region: {target_region} (averaged) in either file. Skipping subplot.")
        ax.set_title(f"{layer.upper()} (No Data)", fontsize=12, color='grey')
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10, color='grey')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        # Set x-axis for "No Data" plots as well for consistent grid
        ax.set_xlim(plot_epoch_min, plot_epoch_max)
        ax.set_xticks(range(plot_epoch_min, plot_epoch_max + 1, epoch_tick_step))
        continue

    # Plot full model line from filtered_df_full
    if not layer_data_full.empty and score_col in layer_data_full.columns and epoch_col in layer_data_full.columns:
        line, = ax.plot(layer_data_full[epoch_col], layer_data_full[score_col],
                        color=full_color, alpha=0.95, linewidth=1.8,
                        label='Full',
                        marker='o', markersize=5, markeredgewidth=1.0,
                        zorder=3)
        if 'Full' not in handles_added:
            legend_handles.append(line)
            legend_labels.append('Full')
            handles_added.add('Full')

    # Plot PCA variants from filtered_df_pca
    if not layer_data_pca.empty and score_col in layer_data_pca.columns and epoch_col in layer_data_pca.columns and pca_n_classes_col in layer_data_pca.columns:
        pca_variants_present = sorted(layer_data_pca[pca_n_classes_col].dropna().unique())

        for n_classes_val in pca_variants_present:
            try:
                n_classes = int(float(n_classes_val)) # Convert safely
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric pca_n_classes value '{n_classes_val}' during plotting for layer {layer}.")
                continue

            layer_pca_variant = layer_data_pca[layer_data_pca[pca_n_classes_col] == n_classes_val] # Filter using original value before conversion for safety

            if not layer_pca_variant.empty:
                color = pca_colors.get(n_classes, '#CCCCCC') # Use converted int for color lookup
                label = f'PCA-{n_classes}'
                line, = ax.plot(layer_pca_variant[epoch_col], layer_pca_variant[score_col],
                                color=color, alpha=0.95, linewidth=1.8,
                                label=label,
                                marker='s', markersize=5, markeredgewidth=1.0,
                                zorder=2)
                if label not in handles_added:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    handles_added.add(label)

    ax.set_title(layer.upper(), fontsize=13, fontweight='medium')
    ax.tick_params(axis='both', which='major', labelsize=10, pad=5, length=4)
    ax.grid(True, linestyle='--', alpha=0.3, color='#666666', zorder=1)
    ax.set_facecolor('white')

    # Set x-axis limits and ticks for this subplot
    ax.set_xlim(plot_epoch_min, plot_epoch_max)
    ax.set_xticks(range(plot_epoch_min, plot_epoch_max + 1, epoch_tick_step))

# Turn off the last unused subplot (2nd row, 4th column)
axes[1, 3].axis('off')

# Add shared X and Y labels
fig.text(0.5, 0.03, 'Epoch', ha='center', va='center', fontsize=14, fontweight='medium')
fig.text(0.04, 0.5, 'RSA Score', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='medium')

# Add legend to the space of the unused subplot
if legend_handles:
    sorted_legend_items = sorted(zip(legend_labels, legend_handles), key=lambda x: float('-inf') if x[0] == 'Full' else int(x[0].split('-')[1]))
    sorted_labels, sorted_handles = zip(*sorted_legend_items)
    fig.legend(sorted_handles, sorted_labels, loc='center', bbox_to_anchor=(0.87, 0.28),
               fontsize=11, framealpha=0.9, edgecolor='none', fancybox=False, title="Model Type")

# Adjust layout
plt.tight_layout(rect=[0.06, 0.05, 0.98, 0.94])

# Save plot
output_dir = 'plots/rsa_temporal_grid'
os.makedirs(output_dir, exist_ok=True)
safe_region_name = target_region.replace(" ", "_").lower()
output_path = os.path.join(output_dir, f'all_layers_temporal_{safe_region_name}_avg_subjects.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)

print(f"Generated grid plot for all layers, {target_region}, Averaged Across Subjects at {output_path}")
print("Script finished.") 