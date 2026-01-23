import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rsm_triangles(npz_path, output_dir="plotters/plots", keys_to_plot=['fc2', 'neural'], percentile_clip=98):
    """
    Loads specified RSMs (default ['fc2', 'neural']) from an .npz file,
    plots their upper triangles side-by-side using matplotlib.pyplot.imshow.
    Styling includes RdBu_r colormap, no axes/box, symmetric off-diagonal scaling
    (clipped to a percentile, default 98th, to handle outliers), and individual colorbars.

    Args:
        npz_path (str): Path to the .npz file containing RSMs.
        output_dir (str): Directory to save the plot image. Created if it doesn't exist.
        keys_to_plot (list): List of keys corresponding to the RSMs to plot.
        percentile_clip (float): Percentile (0-100) to clip the absolute off-diagonal values at
                                 for determining the symmetric color scale (default 98).
    """
    try:
        # Load data from the .npz file
        data = np.load(npz_path)

        # Filter keys that exist in the data
        valid_keys = [key for key in keys_to_plot if key in data]
        if not valid_keys:
            print(f"Error: None of the specified keys {keys_to_plot} found in {npz_path}.")
            available_keys = list(data.keys())
            if available_keys:
                print(f"Available keys: {', '.join(available_keys)}")
            else:
                print("No data keys found in the file.")
            return
        elif len(valid_keys) < len(keys_to_plot):
            print(f"Warning: Could not find keys {[k for k in keys_to_plot if k not in valid_keys]}. Plotting only: {', '.join(valid_keys)}")

        num_plots = len(valid_keys)

        # Determine figure layout
        cols = num_plots
        rows = 1
        fig_width = 6 * cols
        fig_height = 6 * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        print(f"Plotting upper triangles for RSMs: {', '.join(valid_keys)} (scaling clipped to {percentile_clip}th percentile)")

        for i, key in enumerate(valid_keys):
            ax = axes[i]
            rsm_full = data[key]

            if not isinstance(rsm_full, np.ndarray) or rsm_full.ndim != 2 or rsm_full.shape[0] != rsm_full.shape[1]:
                print(f"Warning: Data for key '{key}' is not a square 2D NumPy array. Skipping.")
                ax.axis('off')
                continue

            rsm = rsm_full.copy()
            n_stimuli = rsm.shape[0]
            print(f"  - Processing '{key}' RSM ({n_stimuli}x{n_stimuli})")

            # Check original RSM data
            has_nan_orig = np.isnan(rsm).any()
            print(f"    RSM data: shape={rsm.shape}, original min={np.nanmin(rsm):.4f}, original max={np.nanmax(rsm):.4f}, has_nan={has_nan_orig}")

            # Calculate symmetric vmin/vmax using percentile clipping on off-diagonal absolute values
            rsm_no_diag = rsm.copy()
            np.fill_diagonal(rsm_no_diag, np.nan)
            abs_vals_off_diag = np.abs(rsm_no_diag[~np.isnan(rsm_no_diag)])

            if abs_vals_off_diag.size == 0:
                 print(f"    Warning: No valid off-diagonal values found for '{key}'. Using default scale [-1, 1].")
                 vmin, vmax = -1, 1
            else:
                # Mean-centered scaling for all keys
                mean_val = np.nanmean(rsm_no_diag)
                std_val = np.nanstd(rsm_no_diag)
                # Use mean +/- 2*std for the range, or adjust multiplier as needed
                scale_range = 2 * std_val
                vmin = mean_val - scale_range
                vmax = mean_val + scale_range
                print(f"    Using mean-centered scale for '{key}': mean={mean_val:.4f}, std={std_val:.4f}")
                print(f"    Calculated scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

                # Clip to theoretical Pearson correlation bounds [-1, 1]
                vmin = max(vmin, -1.0)
                vmax = min(vmax, 1.0)
                print(f"    Clipped scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

            # Prepare matrix for plotting: mask lower triangle by setting to NaN
            mask = np.tril(np.ones_like(rsm, dtype=bool))
            rsm_upper_triangle = rsm.copy()
            rsm_upper_triangle[mask] = np.nan

            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad(color='white') # Set NaN values (masked area) to white

            # Plot RSM upper triangle using imshow
            im = ax.imshow(rsm_upper_triangle, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

            # Add colorbar for this subplot with updated label
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation") # Updated label
            # cbar.ax.tick_params(labelsize=10)

            # Remove axes, ticks, labels, and spines to show only the triangle
            ax.axis('off')
            # Add a simple title above the plot area
            if key == 'neural': # Specific title for the 'neural' key
                ax.set_title("RSM of data", fontsize=12, y=1.0)
            else: # Default title for other keys
                ax.set_title(f"RSM: {key}", fontsize=12, y=1.0)


        # Add a main title for the figure - updated title
        fig_title = "" # Remove main title
        if fig_title:
            plt.suptitle(fig_title, fontsize=16, y=1.05)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Ensure output directory exists and save the plot
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(npz_path))[0]
        keys_str = "_".join(valid_keys)
        save_path = os.path.join(output_dir, f"{base_filename}_{keys_str}_rsm_triangles_cbar_p{percentile_clip}_clip_plot.png") # Filename reflects percentile
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Saved RSM triangles plot ({keys_str}, {percentile_clip}% clip) to: {save_path}")

        # plt.show() # Keep commented out unless interactive view needed

    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
    except Exception as e:
        print(f"An error occurred while plotting RSM triangles: {e}")

if __name__ == '__main__':
    rsm_file_path = "model_checkpoints/RSMs/nsd_synthetic/pca4cls/pca_labels_True_cfgid_2_seed_1.npz"
    keys_to_plot_main = ['fc1', 'neural']
    clip_percentile = 98 # Define percentile for clipping (changed to 98)
    if os.path.exists(rsm_file_path):
        plot_rsm_triangles(rsm_file_path, output_dir="plotters/plots", keys_to_plot=keys_to_plot_main, percentile_clip=clip_percentile)
    else:
        print(f"Error: The specified RSM file does not exist: {rsm_file_path}")
        print("Please ensure the path is correct and the file was generated successfully.")
