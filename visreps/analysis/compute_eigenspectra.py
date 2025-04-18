import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import warnings
import glob # Keep glob in case it's useful elsewhere, or remove if strictly not needed.

# Define a default input file path
DEFAULT_INPUT_FILE = "datasets/obj_cls/imagenet-mini-50/features_alexnet_pretrained_none.npz"

def analyze_layer_pca(features):
    """Performs PCA on a single layer's features and returns the raw eigenvalues (eigenspectrum)."""
    if features is None or features.size == 0 or features.ndim != 2:
        warnings.warn(f"Skipping PCA for layer due to invalid features shape or empty data: {features.shape if hasattr(features, 'shape') else 'None'}")
        return None
    if features.shape[0] < 2:
        warnings.warn(f"Skipping PCA for layer: requires at least 2 samples, but got {features.shape[0]}.")
        return None

    # Ensure features are float32 for PCA
    features = features.astype(np.float32)

    # Handle potential NaN/Inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        warnings.warn("Features contain NaN or Inf values. Replacing with 0 before PCA.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        # n_components=None uses min(n_samples, n_features)
        pca = PCA(n_components=None, svd_solver='full')
        pca.fit(features)
        # Return raw eigenvalues instead of the ratio
        eigenvalues = pca.explained_variance_
        return eigenvalues
    except Exception as e:
        warnings.warn(f"PCA failed for layer: {e}")
        return None

def process_file(input_path, output_suffix):
    """Processes a single input npz file to compute and save eigenspectra."""
    print(f"Processing file: {input_path}")

    if not os.path.exists(input_path):
        warnings.warn(f"Input file not found: {input_path}. Skipping.")
        return

    try:
        data = np.load(input_path, allow_pickle=True)
    except Exception as e:
        warnings.warn(f"Failed to load npz file {input_path}: {e}. Skipping.")
        return

    eigenspectra = {}
    # Exclude potential metadata keys more robustly
    layer_keys = [k for k in data.keys() if isinstance(data[k], np.ndarray) and data[k].ndim >= 2]

    if not layer_keys:
        print(f"  Warning: No suitable layer data found in {os.path.basename(input_path)}. Skipping.")
        return

    print(f"  Found layers: {', '.join(layer_keys)}")
    for layer_name in layer_keys:
        print(f"    Analyzing layer: {layer_name}")
        features = data[layer_name]
        spectrum = analyze_layer_pca(features)
        if spectrum is not None:
            eigenspectra[layer_name] = spectrum
            print(f"      -> Eigenvalues shape: {spectrum.shape}")
        else:
            print(f"      -> Skipped PCA.")

    if not eigenspectra:
        print(f"  No eigenspectra were successfully computed for {os.path.basename(input_path)}. No output file will be saved for this input.")
        return

    # Construct output path in the same directory as input
    input_dir = os.path.dirname(input_path)
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    # Avoid adding suffix if the file already has it (e.g., re-running)
    if not base_name.endswith(output_suffix):
        output_filename = f"{base_name}{output_suffix}{ext}"
    else:
        output_filename = f"{base_name}{ext}" # Use original name if suffix already present
    output_path = os.path.join(input_dir, output_filename)

    print(f"  Saving eigenvalues to: {output_path}")
    try:
        np.savez_compressed(output_path, **eigenspectra)
        print("  Eigenvalues saved successfully.")
    except Exception as e:
        print(f"  Error saving output file {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Perform PCA on extracted layer representations from a specific NPZ file and save eigenspectra.")
    parser.add_argument('--input_path', type=str, required=False, default=None,
                        help=f"Path to the .npz representation file. Defaults to '{DEFAULT_INPUT_FILE}' if not provided.")
    parser.add_argument('--output_suffix', type=str, default='_eigenspectra',
                        help="Suffix to add to the input filename for the output file (default: '_eigenspectra').")

    args = parser.parse_args()

    input_path = args.input_path if args.input_path else DEFAULT_INPUT_FILE
    output_suffix = args.output_suffix

    print(f"--- Starting Eigenvalue Analysis for File: {input_path} ---")

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found or is not a file: {input_path}")
        print("--- Eigenvalue Analysis Complete (File not processed) ---")
        return

    # Check if the input file already has the suffix
    base_name = os.path.basename(input_path)
    if base_name.endswith(f"{output_suffix}.npz"):
        print(f"Input file '{base_name}' already seems to be an output file (ends with '{output_suffix}.npz'). Skipping processing.")
        print("--- Eigenvalue Analysis Complete (File skipped) ---")
        return

    # Process the single input file
    process_file(input_path, output_suffix)

    print("--- Eigenvalue Analysis Complete ---")


if __name__ == '__main__':
    main() 