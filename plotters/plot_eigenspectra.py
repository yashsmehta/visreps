import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

# Define standard layer prefixes
CONV_PREFIX = 'conv'
FC_PREFIXES = ['fc', 'linear'] # Allow for 'fc' or 'linear'

# Define default input files if none are provided via CLI
DEFAULT_INPUT_PATHS = [
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_0_eigenspectra.npz",
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_5_eigenspectra.npz",
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_10_eigenspectra.npz",
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_15_eigenspectra.npz",
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_20_eigenspectra.npz",
    "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_25_eigenspectra.npz",
]

def calculate_eff_dim(eigenvalues):
    """Calculates the Effective Dimensionality (Participation Ratio)."""
    if eigenvalues is None or eigenvalues.size == 0:
        return None
    # Ensure eigenvalues are positive for calculation stability if needed
    eigenvalues = eigenvalues[eigenvalues > 1e-12] # Avoid tiny/zero eigenvalues causing issues
    if eigenvalues.size == 0:
        return 0.0 # Or handle as appropriate
    try:
        sum_l = np.sum(eigenvalues)
        sum_l_sq = np.sum(eigenvalues**2)
        if sum_l_sq < 1e-12: # Avoid division by zero
            return 0.0
        eff_dim = (sum_l**2) / sum_l_sq
        return eff_dim
    except Exception as e:
        warnings.warn(f"Could not calculate Eff Dim: {e}")
        return None

def plot_layer_subplots(spectra_data, figure_title, output_path, input_labels):
    """Creates a figure with horizontal subplots for each layer, comparing multiple input spectra."""
    layer_names = sorted(spectra_data.keys())
    n_layers = len(layer_names)
    n_inputs = len(input_labels)

    if n_layers == 0:
        print(f"No layers found for figure: {figure_title}. Skipping plot generation.")
        return
    if n_inputs == 0:
        print(f"No input files provided for figure: {figure_title}. Skipping plot generation.")
        return

    # Adjust figure size based on the number of subplots
    fig_width = max(8, n_layers * 3.5)
    fig_height = 5 + n_inputs * 0.1 # Slightly increase height for more text

    # Use Seaborn context manager for consistent styling
    with sns.axes_style("ticks"): # Removed plt.style.context for consistency
        # Create subplots: 1 row, n_layers columns, share y-axis
        fig, axes = plt.subplots(1, n_layers, figsize=(fig_width, fig_height), sharey=True)

        # If only one subplot, axes is not an array, make it iterable
        if n_layers == 1:
            axes = [axes]

        # Define colors and linestyles for multiple inputs
        colors = sns.color_palette("viridis", n_inputs)
        linestyles = ['-', '--', ':', '-.'] * (n_inputs // 4 + 1)

        all_handles = []
        all_labels = []

        for i, layer_name in enumerate(layer_names):
            ax = axes[i]
            layer_spectra = spectra_data[layer_name] # Dict: {input_label: spectrum}

            eff_dims = {}
            # Plot spectrum for each input file
            for j, label in enumerate(input_labels):
                spectrum = layer_spectra.get(label)
                if spectrum is not None and spectrum.size > 0:
                    ranks = np.arange(1, len(spectrum) + 1)
                    line, = ax.plot(ranks, spectrum, marker='.', markersize=4, linestyle=linestyles[j],
                                  color=colors[j], label=label)
                    if i == 0: # Collect handles/labels only from the first subplot
                        all_handles.append(line)
                        all_labels.append(label)

                    # Calculate Eff Dim for this spectrum
                    eff_dims[label] = calculate_eff_dim(spectrum)
                else:
                    eff_dims[label] = None

            # Define bbox properties for text background
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75)

            # Display Eff Dim values on the plot with white background
            text_y_pos = 0.95
            text_x_pos = 0.95
            text_spacing = 0.09
            for j, label in enumerate(input_labels):
                eff_dim = eff_dims.get(label)
                if eff_dim is not None:
                    # Consistent label shortening for display text
                    if label == "Untrained":
                        short_label = "Untrained"
                    elif label == "1K Classes (ImageNet)":
                        short_label = "1K Classes"
                    elif " Classes" in label: # Handle "X Classes" labels
                        short_label = label # Display as '2 Classes', '4 Classes', ...
                    else: # Fallback for unexpected labels (like raw cfgX if mapping failed)
                        short_label = label # Use the original label

                    eff_dim_text = r'{} Eff Dim: $\mathbf{{{:.1f}}}$'.format(short_label, eff_dim)
                    ax.text(text_x_pos, text_y_pos, eff_dim_text,
                            transform=ax.transAxes, fontsize=8, color=colors[j],
                            verticalalignment='top', horizontalalignment='right',
                            bbox=bbox_props)
                    text_y_pos -= text_spacing # Adjust spacing

            ax.set_title(layer_name)
            ax.set_yscale('log') # Set y-axis to log scale
            ax.set_ylim(bottom=1e-10) # Set lower y-limit
            ax.grid(True, which="both", ls="--", alpha=0.6)
            sns.despine(ax=ax)

            # Set x-axis label only for the middle plot(s) for clarity
            if i == n_layers // 2:
                ax.set_xlabel("Rank")

        # Set shared y-axis label only on the first subplot
        axes[0].set_ylabel(r'$\log(\lambda_i)$')

        # Add an overall figure title
        fig.suptitle(figure_title, y=1.02) # Adjust y position to avoid overlap

        # Create a shared legend below the subplots
        if all_handles: # Only create legend if there are lines plotted
            # Consistent shortening for legend
            short_labels = []
            for l in all_labels: # l might be 'Epoch 0', 'Epoch 5', etc.
                if l == "Untrained":
                    short_labels.append("Untrained")
                elif l == "1K Classes (ImageNet)":
                    short_labels.append("1K Classes")
                elif " Classes" in l:
                    short_labels.append(l) # Use '2 Classes', '4 Classes', ...
                else: # Fallback (e.g., for epoch labels)
                     short_labels.append(l) # Use original label if no known pattern matches

            # Adjust bbox_to_anchor y-value to reduce space (e.g., from -0.1 to -0.05)
            fig.legend(all_handles, short_labels, loc='lower center', ncol=min(n_inputs, 4), bbox_to_anchor=(0.5, -0.05), fontsize='medium')

        # Adjust layout to prevent labels/titles overlapping and make space for legend
        # Keep rect relatively tight if legend moved up
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect bottom slightly

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    -> Subplot figure saved to: {os.path.basename(output_path)}")
    except Exception as e:
        warnings.warn(f"Failed to save subplot figure to {output_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

def main():
    # Set Seaborn global theme for a polished look
    sns.set_theme(style="ticks", context="paper")

    parser = argparse.ArgumentParser(description="Plot grouped eigenspectra comparison from multiple input NPZ files.")
    parser.add_argument('--input_paths', type=str, nargs='*', required=False, default=None, # Changed nargs to '*' and default to None
                        help="One or more paths to the input eigenspectra .npz files. If not provided, uses default paths defined in the script.")
    parser.add_argument('--labels', type=str, nargs='+', required=False, default=None,
                        help="Optional labels for each input path (must match the number of input paths). Defaults to filenames.")
    parser.add_argument('--output_dir', type=str, default='plotters/plots/eigenspectra_comparison',
                        help="Base directory to save output plots. Default: 'plotters/plots/eigenspectra_comparison'.")
    parser.add_argument('--output_prefix', type=str, default='comparison',
                        help="Prefix for the output plot filenames (default: 'comparison').")

    args = parser.parse_args()

    # Use default paths if none are provided via CLI
    input_paths = args.input_paths if args.input_paths else DEFAULT_INPUT_PATHS

    # Validate labels
    input_labels = args.labels
    if input_labels:
        if len(input_labels) != len(input_paths):
            raise ValueError("Number of labels must match the number of input paths being used (provided or default).")
    else:
        # Default label generation: Extract unique part (e.g., 'cfgX') or use fallback
        input_labels = []
        for p in input_paths:
            # Attempt to extract 'cfgX' part from the path
            parts = p.split(os.sep)
            cfg_part = next((part for part in reversed(parts) if part.startswith('cfg')), None)

            generated_label = None
            # Priority 1: Check for epoch number in filename
            filename = os.path.basename(p)
            epoch_match = re.search(r'_epoch_(\d+)_eigenspectra\.npz', filename)
            if epoch_match:
                epoch_num = epoch_match.group(1)
                generated_label = f"Epoch {epoch_num}"

            # Priority 2: Check for special keywords if epoch not found
            if generated_label is None:
                if 'none' in p.lower():
                    generated_label = "Untrained"
                elif 'imagenet1k' in p.lower():
                    generated_label = "1K Classes (ImageNet)"

            # Priority 3: Map cfgX to PCA classes if still no label (less likely relevant now)
            if generated_label is None and cfg_part:
                try:
                    cfg_num = int(cfg_part[3:])
                    num_classes = 2**cfg_num
                    generated_label = f"{num_classes} Classes"
                except (ValueError, IndexError):
                    pass

            # Priority 4: Fallback to base filename without extension
            if generated_label is None:
                generated_label = os.path.splitext(filename)[0].replace('_eigenspectra', '')

            input_labels.append(generated_label)

    # Load data from all specified paths
    all_eigenspectra_data = {}
    all_layer_keys = set()

    print("Loading input files...")
    for i, input_path in enumerate(input_paths):
        label = input_labels[i]
        if not os.path.exists(input_path):
            warnings.warn(f"Input eigenspectra file not found: {input_path}. Skipping.")
            continue

        print(f"  Loading: {input_path} (Label: '{label}')")
        try:
            data = np.load(input_path)
            all_eigenspectra_data[label] = data
            all_layer_keys.update(data.keys())
        except Exception as e:
            warnings.warn(f"Failed to load npz file {input_path}: {e}. Skipping.")

    if not all_eigenspectra_data:
        print("Error: Failed to load any valid eigenspectra data. Exiting.")
        return

    all_layer_keys = sorted(list(all_layer_keys))

    if not all_layer_keys:
        print("Warning: No layer data found in the loaded files. Nothing to plot.")
        return

    # Group spectra by layer type (conv vs fc)
    conv_spectra = {}
    fc_spectra = {}

    print(f"Processing layers: {', '.join(all_layer_keys)}")
    for layer_name in all_layer_keys:
        layer_data_across_inputs = {}
        has_data_for_layer = False
        for label, data in all_eigenspectra_data.items():
            spectrum = data.get(layer_name)
            layer_data_across_inputs[label] = spectrum
            if spectrum is not None:
                has_data_for_layer = True

        if not has_data_for_layer:
            # warnings.warn(f"Layer '{layer_name}' has no data across any input. Skipping.")
            continue # Skip layer if no input file contains it

        if layer_name.startswith(CONV_PREFIX):
            print(f"  -> Grouping '{layer_name}' as Convolutional")
            conv_spectra[layer_name] = layer_data_across_inputs
        elif any(layer_name.startswith(prefix) for prefix in FC_PREFIXES):
            print(f"  -> Grouping '{layer_name}' as Fully Connected")
            fc_spectra[layer_name] = layer_data_across_inputs
        else:
            warnings.warn(f"Layer '{layer_name}' does not match known prefixes ('{CONV_PREFIX}', {FC_PREFIXES}). Skipping.")

    # Construct full output directory path
    # Simplified output path - just use output_dir directly
    full_output_dir = args.output_dir
    os.makedirs(full_output_dir, exist_ok=True)
    print(f"Saving plots to directory: {full_output_dir}")

    # --- Plotting Layer Subplots --- #

    # Plot Convolutional Layers Subplots
    if conv_spectra:
        conv_title = f"Convolutional Layer Eigenvalue Evolution Across Epochs"
        conv_output_filename = f"{args.output_prefix}_conv_layers_subplots.png"
        conv_output_path = os.path.join(full_output_dir, conv_output_filename)
        print("Plotting Convolutional Layer Subplots...")
        plot_layer_subplots(conv_spectra, conv_title, conv_output_path, input_labels)
    else:
        print("No convolutional layers found to plot.")

    # Plot Fully Connected Layers Subplots
    if fc_spectra:
        fc_title = f"Fully Connected Layer Eigenvalue Evolution Across Epochs"
        fc_output_filename = f"{args.output_prefix}_fc_layers_subplots.png"
        fc_output_path = os.path.join(full_output_dir, fc_output_filename)
        print("Plotting Fully Connected Layer Subplots...")
        plot_layer_subplots(fc_spectra, fc_title, fc_output_path, input_labels)
    else:
        print("No fully connected layers found to plot.")

    print("--- Subplot Eigenspectra Comparison Plotting Complete ---")

if __name__ == '__main__':
    # Optional: Set backend for non-interactive environments if needed
    # import matplotlib
    # matplotlib.use('Agg')
    main() 