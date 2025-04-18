import os
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings

# Define a default input file path (using the example provided by the user)
DEFAULT_INPUT_FILE = "model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_25.npz"

def twoNN_id(X, *, decimate=[1, 2, 5, 10], n_jobs=-1, verbose=False):
    """
    Estimate the intrinsic dimensionality of a data‐set with the TwoNN method
    (Facco et al. 2017; Ansuini et al. 2019).

    Parameters
    ----------
    X : (N, D) numpy.ndarray
        Data matrix (activations). Rows = samples, columns = features.
        Must be float32/float64.
    decimate : list[int], optional
        k‐fold factors for the scale‐stability check. For each k the ID is recomputed
        on a 1/k random subsample. Use [1] to skip the check. Default: [1, 2, 5, 10].
    n_jobs : int, optional
        Threads used by scikit‑learn's NearestNeighbors. Default: -1 (use all available cores).
    verbose : bool, optional
        Print progress messages. Default: False.

    Returns
    -------
    id_full : float
        ID estimate on the full data (k=1).
    id_by_scale : dict[int, float]
        ID estimates on each decimation level {k : id_k}. Returns None if calculation fails.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        warnings.warn(f"Input data must be a 2D numpy array. Got shape: {X.shape if hasattr(X, 'shape') else type(X)}")
        return None, None
    if X.shape[0] < 3: # Need at least 3 points to find 2 nearest neighbors
        warnings.warn(f"Skipping TwoNN ID estimation: requires at least 3 samples, but got {X.shape[0]}.")
        return None, None

    # Ensure features are float type for distance calculations
    if not np.issubdtype(X.dtype, np.floating):
         warnings.warn(f"Input data type is {X.dtype}, converting to float32.")
         X = X.astype(np.float32)

    # Handle potential NaN/Inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        warnings.warn("Features contain NaN or Inf values. Replacing with 0 before TwoNN.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng()
    id_by_scale = {}

    try:
        # pre‑build NN structure once – reused for all scales ≤ original N
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto',
                                metric='euclidean', n_jobs=n_jobs)
        nbrs.fit(X)

        for k in sorted(list(set(decimate))): # Ensure unique, sorted decimation factors
            if k <= 0:
                warnings.warn(f"Decimation factor k must be > 0, skipping k={k}.")
                continue

            num_samples_sub = X.shape[0] // k
            if num_samples_sub < 3:
                 warnings.warn(f"Skipping decimation k={k}: results in {num_samples_sub} samples, needs >= 3.")
                 id_by_scale[k] = np.nan # Use NaN to indicate failure for this scale
                 continue

            if k == 1:
                X_sub = X
                idx_sub = slice(None) # Represents all indices
            else:
                # 1 / k random subsample *without replacement*
                idx_sub = rng.choice(X.shape[0], num_samples_sub, replace=False)
                X_sub = X[idx_sub]

            if verbose:
                print(f'  Calculating ID for k={k} (N = {X_sub.shape[0]:,})')

            # query on the (possibly smaller) set
            # Important: query neighbors within the *original* full set `X` using the pre-built `nbrs`
            # This is crucial for consistent neighbor finding across scales.
            # We find neighbors for the points in X_sub within the full dataset X.
            dists, indices = nbrs.kneighbors(X_sub, n_neighbors=3, return_distance=True)

            # Check if any point's nearest neighbors include itself multiple times (can happen with duplicates)
            valid_neighbors = indices[:, 1] != indices[:, 0] # Check 1st NN is not the point itself (at index 0)
            valid_neighbors &= indices[:, 2] != indices[:, 0] # Check 2nd NN is not the point itself

            # Filter distances based on valid neighbors
            r1 = dists[valid_neighbors, 1]
            r2 = dists[valid_neighbors, 2]

            # Avoid numerical issues (r1=0 implies duplicate points)
            mask = r1 > 1e-12 # Use a small epsilon instead of > 0
            if not np.any(mask):
                 warnings.warn(f"Skipping ID calculation for k={k}: no valid neighbor pairs with r1 > 0 found (possibly many duplicate points).")
                 id_by_scale[k] = np.nan
                 continue

            r1_masked = r1[mask]
            r2_masked = r2[mask]

            # Further avoid r2=0 if r1 > 0 (can happen with >2 identical points)
            mask_r2 = r2_masked > 1e-12
            if not np.any(mask_r2):
                warnings.warn(f"Skipping ID calculation for k={k}: no valid neighbor pairs with r2 > 0 found (possibly many duplicate points).")
                id_by_scale[k] = np.nan
                continue

            mu = r2_masked[mask_r2] / r1_masked[mask_r2]

            # Avoid log(0) issues, though masks should handle this
            mu = mu[mu > 1e-12]
            if mu.size == 0:
                warnings.warn(f"Skipping ID calculation for k={k}: no valid mu values > 0 after filtering.")
                id_by_scale[k] = np.nan
                continue

            # MLE:   d  =  1 / mean(log mu)
            log_mu = np.log(mu)
            if np.isinf(np.mean(log_mu)) or np.isnan(np.mean(log_mu)) or np.mean(log_mu) == 0:
                 warnings.warn(f"Problem calculating mean(log(mu)) for k={k}. Result is {np.mean(log_mu)}. Setting ID to NaN.")
                 d_hat = np.nan
            else:
                 d_hat = 1.0 / np.mean(log_mu)

            id_by_scale[k] = d_hat
            if verbose:
                print(f'    -> ID ≈ {d_hat:.2f}')

    except Exception as e:
        warnings.warn(f"Error during TwoNN ID estimation: {e}")
        # Try to return partial results if available, otherwise None
        if not id_by_scale:
            return None, None
        else:
             # Ensure all requested scales are present, filling failures with NaN
             for k_req in decimate:
                 if k_req not in id_by_scale:
                     id_by_scale[k_req] = np.nan
             id_full = id_by_scale.get(1, np.nan) # Get ID for k=1 if available
             return id_full, id_by_scale


    id_full = id_by_scale.get(1, np.nan) # Get ID for k=1, default to NaN if not computed
    # Ensure all requested decimation factors are keys in the output, even if failed (NaN)
    for k_req in decimate:
        if k_req not in id_by_scale:
            id_by_scale[k_req] = np.nan

    return id_full, id_by_scale


def analyze_layer_twoNN(features, decimate_factors, n_jobs, verbose):
    """Computes TwoNN ID for a single layer's features and assesses stability."""
    if features is None or not isinstance(features, np.ndarray) or features.ndim != 2:
        warnings.warn(f"Skipping TwoNN for layer due to invalid features shape or non-array data: {features.shape if hasattr(features, 'shape') else type(features)}")
        return None, None # Return None for both ID and variation
    if features.shape[0] < 3:
        warnings.warn(f"Skipping TwoNN for layer: requires at least 3 samples, but got {features.shape[0]}.")
        return None, None

    representative_id = None
    max_variation_pct = None

    try:
        id_full, id_by_scale = twoNN_id(features, decimate=decimate_factors, n_jobs=n_jobs, verbose=verbose)
        
        # Check if primary ID calculation failed
        if id_full is None or np.isnan(id_full):
             warnings.warn("Primary ID (k=1) calculation failed or resulted in NaN.")
             return None, None # Cannot determine representative ID or stability

        representative_id = id_full # Per guideline, always use k=1 if stable enough
        
        # Calculate stability (max variation percentage)
        valid_scale_ids = {k: v for k, v in id_by_scale.items() if v is not None and not np.isnan(v) and k > 1}
        
        if not valid_scale_ids or representative_id == 0: # No other scales to compare or division by zero
            max_variation_pct = 0.0
        else:
            deviations = [abs(v - representative_id) / representative_id for k, v in valid_scale_ids.items()]
            max_variation_pct = max(deviations) * 100.0 if deviations else 0.0
            
        # Optional: Add check based on variation threshold if needed later
        # if max_variation_pct > 10.0:
        #     warnings.warn(f"High variation ({max_variation_pct:.1f}%) observed. ID estimate might be less reliable.")
            
        return representative_id, max_variation_pct
        
    except Exception as e:
        warnings.warn(f"TwoNN ID calculation or stability analysis failed for layer: {e}")
        return None, None

def process_file(input_path, output_suffix, decimate_factors, n_jobs):
    """Processes a single input npz file to compute and save TwoNN intrinsic dimensionalities."""
    print(f"Processing file: {input_path}")

    if not os.path.exists(input_path):
        warnings.warn(f"Input file not found: {input_path}. Skipping.")
        return

    try:
        # Load the data, allowing pickles in case metadata is stored
        with np.load(input_path, allow_pickle=True) as data:
            # Robustly identify layer keys (assuming they are multi-dimensional numerical arrays)
            # --- MODIFIED: Specifically target 'conv*' and 'fc*' layers --- 
            potential_layer_keys = [k for k in data.files if isinstance(data[k], np.ndarray) and data[k].ndim >= 1]
            layer_keys = [k for k in potential_layer_keys if k.startswith('conv') or k.startswith('fc')]
            # --- END MODIFICATION --- 

            # Try to handle potential metadata stored as 0-d arrays or non-array objects
            metadata_keys = [k for k in data.files if k not in layer_keys]
            if metadata_keys:
                print(f"  Ignoring non-layer keys: {', '.join(metadata_keys)}")

            if not layer_keys:
                print(f"  Warning: No suitable layer data (numpy arrays starting with 'conv' or 'fc') found in {os.path.basename(input_path)}. Skipping.")
                return

            print(f"  Found layers for analysis: {', '.join(layer_keys)}")
            layer_ids = {}

            for layer_name in layer_keys:
                print(f"    Analyzing layer: {layer_name}")
                features = data[layer_name]
                print(f"      Original features shape: {features.shape}") # Print original shape

                # --- Skip non-numeric data (Redundant with new layer selection, but harmless) ---
                if not np.issubdtype(features.dtype, np.number):
                    print(f"      Skipping layer '{layer_name}': Non-numeric data type ({features.dtype}).")
                    continue
                # --- End Skip ---

                # Reshape if it's a flat array (e.g., (N,)) which might be valid for some layers
                if features.ndim == 1:
                    warnings.warn(f"Layer '{layer_name}' has 1 dimension ({features.shape}). Reshaping to ({features.shape[0]}, 1) for TwoNN.")
                    features = features.reshape(-1, 1)
                elif features.ndim > 2:
                     # Flatten features if they are like (N, H, W, C) -> (N, H*W*C)
                     original_shape = features.shape
                     features = features.reshape(original_shape[0], -1)
                     print(f"      Flattened layer '{layer_name}' from {original_shape} to {features.shape}")


                # --- MODIFIED: Get representative ID and variation ---
                representative_id, max_variation_pct = analyze_layer_twoNN(features, decimate_factors, n_jobs, verbose=False) # Keep verbose=False for analyze_layer

                if representative_id is not None: # Check only representative_id, variation is secondary info
                    # Store results with a simpler key
                    layer_ids[f"{layer_name}_id"] = representative_id
                    # Print representative ID and variation
                    print(f"      -> Representative ID (k=1): {representative_id:.2f} (Max variation: {max_variation_pct:.1f}%)")
                else:
                    print(f"      -> Skipped TwoNN ID calculation or calculation failed.")
                # --- END MODIFICATION ---

    except Exception as e:
        warnings.warn(f"Failed to load or process npz file {input_path}: {e}. Skipping.")

    if not layer_ids:
        print(f"  No TwoNN IDs were successfully computed for {os.path.basename(input_path)}. No output file will be saved.")
        return

    # Construct output path in the same directory as input
    input_dir = os.path.dirname(input_path)
    base_name, ext = os.path.splitext(os.path.basename(input_path))

    # Ensure the extension is .npz
    if ext.lower() != '.npz':
        warnings.warn(f"Input file {base_name}{ext} does not have '.npz' extension. Output will still use '.npz'.")
        ext = '.npz'


    # Avoid adding suffix if the file already has it (e.g., re-running)
    if not base_name.endswith(output_suffix):
        output_filename = f"{base_name}{output_suffix}{ext}"
    else:
        output_filename = f"{base_name}{ext}" # Use original name if suffix already present
    output_path = os.path.join(input_dir, output_filename)

    print(f"  Saving TwoNN IDs to: {output_path}")
    try:
        # Use savez_compressed for potentially smaller file size
        # Allow pickles for saving the dictionary id_by_scale
        np.savez_compressed(output_path, **layer_ids)
        print("  TwoNN IDs saved successfully.")
    except Exception as e:
        print(f"  Error saving output file {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compute TwoNN intrinsic dimensionality (ID) on layer representations from an NPZ file.")
    parser.add_argument('--input_path', type=str, required=False, default=None,
                        help=f"Path to the input .npz representation file. Defaults to '{DEFAULT_INPUT_FILE}' if not provided.")
    parser.add_argument('--output_suffix', type=str, default='_twoNN_id',
                        help="Suffix to add to the input filename for the output file (default: '_twoNN_id').")
    parser.add_argument('--decimate', type=str, default='1,2,5,10',
                        help="Comma-separated list of integers for k-fold decimation factors for scale stability check (e.g., '1,2,4,8'). Default: '1,2,5,10'.")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of parallel jobs for NearestNeighbors (-1 uses all cores). Default: -1.")

    args = parser.parse_args()

    input_path = args.input_path if args.input_path else DEFAULT_INPUT_FILE
    output_suffix = args.output_suffix
    n_jobs = args.n_jobs

    try:
        decimate_factors = [int(k) for k in args.decimate.split(',') if k.strip()]
        if not decimate_factors or any(k <= 0 for k in decimate_factors):
             raise ValueError("Decimation factors must be positive integers.")
        if 1 not in decimate_factors:
            print("Warning: k=1 (full dataset) not included in decimation factors. Adding it.")
            decimate_factors.append(1)
            decimate_factors.sort()
    except ValueError as e:
        print(f"Error: Invalid decimation factors '{args.decimate}'. {e}")
        return

    print(f"--- Starting TwoNN ID Analysis for File: {input_path} ---")
    print(f"Using decimation factors: {decimate_factors}")

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found or is not a file: {input_path}")
        print("--- TwoNN ID Analysis Complete (File not processed) ---")
        return

    # Check if the input file already has the suffix to avoid processing output files
    base_name = os.path.basename(input_path)
    # Ensure we check against the correct extension too
    expected_output_filename = f"{os.path.splitext(base_name)[0]}{output_suffix}.npz"
    # Simple check if basename *ends with* the suffix + .npz
    if base_name.endswith(f"{output_suffix}.npz"):
         print(f"Input file '{base_name}' already seems to be an output file (ends with '{output_suffix}.npz'). Skipping processing.")
         print("--- TwoNN ID Analysis Complete (File skipped) ---")
         return


    # Process the single input file
    process_file(input_path, output_suffix, decimate_factors, n_jobs)

    print("--- TwoNN ID Analysis Complete ---")


if __name__ == '__main__':
    main()
