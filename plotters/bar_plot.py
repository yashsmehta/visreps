import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import patches
# from matplotlib.ticker import MaxNLocator # No longer needed
import pandas as pd
import numpy as np
import os # Add os import

def plot_brain_score_barplot(categories, values, output_filename):
    """
    Generates and saves a bar plot showing brain similarity scores for different training conditions.

    Args:
        categories (list[str]): A list of strings representing the training conditions (categories for the x-axis).
        values (list[float]): A list of floats representing the brain similarity scores corresponding to the categories.
        output_filename (str): The path and filename where the plot will be saved (e.g., "brain_similarity.png").
    """
    data = pd.DataFrame({
        'Training Condition': categories,
        'Brain Similarity': values
    })

    untrained_color = '#AAAAAA'
    # Adjust color palette generation if needed based on the number of categories
    num_pca_cats = len(categories) - 2 if len(categories) > 2 else 0
    num_class_colors = min(num_pca_cats, 6)
    if num_class_colors > 0:
        class_colors = sns.color_palette("Blues", n_colors=num_class_colors + 1)[1:]
    else:
        class_colors = []
    thousand_color = '#FFB74D'

    custom_palette = []
    if "Untrained" in categories:
        custom_palette.append(untrained_color)
    if num_pca_cats > 0:
        custom_palette.extend(list(class_colors))
        # Repeat last color if more pca cats than colors
        if len(custom_palette) - (1 if "Untrained" in categories else 0) < num_pca_cats:
             custom_palette.extend([class_colors[-1]] * (num_pca_cats - (len(custom_palette) - (1 if "Untrained" in categories else 0))))
    if "1000 Classes" in categories:
        custom_palette.append(thousand_color)

    # Handle cases with fewer categories than expected palette length
    if len(custom_palette) > len(categories):
        custom_palette = custom_palette[:len(categories)]
    elif len(custom_palette) < len(categories):
        # Add default colors if palette is too short (edge case)
        default_color = '#CCCCCC'
        custom_palette.extend([default_color] * (len(categories) - len(custom_palette)))


    sns.set_theme(style="white", context="paper", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted figsize slightly for labels

    bar_width = 0.7
    num_categories = len(categories)
    bar_positions = np.arange(num_categories)

    # Adjust hatch patterns based on actual categories present
    hatch_patterns = []
    if "Untrained" in categories:
        hatch_patterns.append('')
    num_middle_hatches = len(categories) - (1 if "Untrained" in categories else 0) - (1 if "1000 Classes" in categories else 0)
    hatch_patterns.extend(['/'] * num_middle_hatches)
    if "1000 Classes" in categories:
        hatch_patterns.append('')

    # Ensure hatch_patterns length matches categories length
    if len(hatch_patterns) < num_categories:
        hatch_patterns.extend([''] * (num_categories - len(hatch_patterns)))
    elif len(hatch_patterns) > num_categories:
        hatch_patterns = hatch_patterns[:num_categories]


    max_val = np.max(values) if values else 0
    # min_val = np.min(values) if values else 0 # Not used
    # plot_range = max_val - 0 # Not used
    # label_space = plot_range * 0.1 if plot_range > 0 else 0.1 # Using fixed ylim

    # bar_coords_and_vals = [] # Not used

    for i, (cat, val) in enumerate(zip(categories, values)):
        x0 = bar_positions[i] - bar_width / 2
        y0 = 0
        width = bar_width
        height = val

        # bar_top_center_x = x0 + width / 2 # Not used
        # bar_top_y = height # Not used
        # bar_coords_and_vals.append(((bar_top_center_x, bar_top_y), val)) # Not used

        boxstyle = mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.1)

        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=boxstyle,
            facecolor=custom_palette[i],
            edgecolor='black',
            linewidth=0.8,
            hatch=hatch_patterns[i],
            mutation_aspect=0.05
        )
        ax.add_patch(rect)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9) # Add labels and rotate
    ax.tick_params(axis='x', which='major', bottom=True, length=4, color='black') # Show x ticks
    ax.tick_params(axis='y', which='major', left=True, labelsize=10, length=4, color='black')

    # Set specific y-ticks
    y_ticks = np.arange(0.0, 0.51, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])

    ax.set_ylim(0, 0.55) # Adjust ylim to accommodate ticks + small margin
    ax.set_xlim(-0.5, num_categories - 0.5)

    # Add y-axis label
    ax.set_ylabel('Brain Similarity (Score)', fontsize=11)

    plt.tight_layout(pad=0.8) # Increase padding slightly

    sns.despine(right=True, top=True)

    # Save as PNG
    plt.savefig(output_filename, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Plot saved as {output_filename}")


# --- Example Usage ---
if __name__ == "__main__":
    base_log_path = 'logs/eval/checkpoint'
    full_model_filename = 'imagenet_cnn.csv'
    pca_filename = 'imagenet_pca.csv'
    layer_to_plot = 'fc2' # Use lowercase to match CSV
    pca_classes_to_plot = [2, 4, 8, 16, 32, 64]

    full_model_path = os.path.join(base_log_path, full_model_filename)
    pca_path = os.path.join(base_log_path, pca_filename)

    # Load the data
    try:
        cnn_data = pd.read_csv(full_model_path)
        pca_data = pd.read_csv(pca_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file: {e.filename}")
        print("Please ensure the CSV files are present at the specified path.")
        exit()

    # Ensure 'layer' column is lowercase for consistent matching
    cnn_data['layer'] = cnn_data['layer'].str.lower()
    pca_data['layer'] = pca_data['layer'].str.lower()

    # --- Extract Untrained Score ---
    untrained_score_series = cnn_data[(cnn_data['layer'] == layer_to_plot) & (cnn_data['epoch'] == 0)]['score']
    if untrained_score_series.empty:
        print(f"Error: Could not find score for layer '{layer_to_plot}' and epoch 0 in {full_model_filename}")
        untrained_score = 0 # Default or handle error
    else:
        untrained_score = untrained_score_series.iloc[0]

    # --- Extract PCA Scores ---
    pca_scores_data = pca_data[
        (pca_data['layer'] == layer_to_plot) &
        (pca_data['pca_n_classes'].isin(pca_classes_to_plot))
    ]
    # Assuming the relevant score for each class is the one from the *last* epoch for that class training
    # Group by pca_n_classes and get the score corresponding to the max epoch within each group
    if not pca_scores_data.empty:
        pca_scores_final_epoch = pca_scores_data.loc[pca_scores_data.groupby('pca_n_classes')['epoch'].idxmax()]
        pca_scores_final_epoch = pca_scores_final_epoch.sort_values('pca_n_classes')
        pca_values = pca_scores_final_epoch['score'].tolist()
        pca_categories = [f"{n} Classes" for n in pca_scores_final_epoch['pca_n_classes'].tolist()]

        if len(pca_scores_final_epoch) != len(pca_classes_to_plot):
             print(f"Warning: Missing PCA scores for some classes in {pca_classes_to_plot} for layer '{layer_to_plot}'. Found {len(pca_scores_final_epoch)} scores.")
    else:
        print(f"Warning: No PCA scores found for layer '{layer_to_plot}' and classes {pca_classes_to_plot}.")
        pca_values = []
        pca_categories = []


    # --- Extract 1000 Classes (Trained) Score ---
    # Assuming 1000 classes corresponds to the standard imagenet training (not pca)
    # Find the score for the latest epoch in the imagenet_cnn data
    cnn_layer_data = cnn_data[cnn_data['layer'] == layer_to_plot]
    if not cnn_layer_data.empty:
        latest_epoch = cnn_layer_data['epoch'].max()
        trained_score_series = cnn_layer_data[cnn_layer_data['epoch'] == latest_epoch]['score']
        if trained_score_series.empty:
             print(f"Error: Could not find score for layer '{layer_to_plot}' and latest epoch {latest_epoch} in {full_model_filename}")
             trained_score = 0 # Default or handle error
        else:
            trained_score = trained_score_series.iloc[0]
    else:
        print(f"Warning: No CNN data found for layer '{layer_to_plot}'.")
        trained_score = 0


    # --- Combine data for plotting ---
    categories = ["Untrained"] + pca_categories + ["1000 Classes"]
    values = [untrained_score] + pca_values + [trained_score]

    # Define output file
    output_dir = "plotters/plots"
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    output_file = os.path.join(output_dir, f"brain_score_barplot_{layer_to_plot}.png") # Change extension to .png

    # Plot
    plot_brain_score_barplot(categories, values, output_file)
