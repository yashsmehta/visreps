import os
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Sequence
import gc
import pickle # Import standard pickle library
import argparse # Add argparse
from PIL import Image # Import PIL Image
import matplotlib.pyplot as plt # Import matplotlib

import visreps.utils as utils
from bonner.datasets.gifford2025_nsd_synthetic._data import load_betas
# Import stimulus loading functions
from bonner.datasets.gifford2025_nsd_synthetic._stimuli import load_shared_stimuli, StimulusSet, N_STIMULI, N_STIMULI_SHARED # Import StimulusSet and N_STIMULI and N_STIMULI_SHARED
from bonner.datasets.allen2021_natural_scenes import load_rois, create_roi_selector


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the specific ROIs we want to process and their selectors
# Based on _roi_helper.py structure for visual streams
SPECIFIC_ROIS: Dict[str, Sequence[Dict[str, str]]] = {
    "early": ({"source": "streams", "label": "early"},),
    "midventral": ({"source": "streams", "label": "midventral"},),
    "ventral": ({"source": "streams", "label": "ventral"},),
}

# Inverse mapping from short name to descriptive name for saving
DESCRIPTIVE_ROI_NAMES = {
    "early": "early visual stream",
    "midventral": "midventral visual stream",
    "ventral": "ventral visual stream",
}

def process_fmri_data(output_path: Path, output_filename: str):
    """Processes and saves fMRI data."""
    logger.info("--- Starting fMRI Data Processing ---")
    # --- Hardcoded Configuration for fMRI ---
    rois_to_process = ["early", "midventral", "ventral"] # Short names used for processing logic
    resolution = "1pt8mm"
    preprocessing = "fithrf_GLMdenoise_RR"
    z_score = True
    # subjects_to_process = [0] # Process only subject 0 for now
    subjects_to_process = list(range(8)) # Process all subjects
    # ---------------------------------------

    # Initialize based on DESCRIPTIVE names for final output structure
    all_processed_data = {name: {} for name in DESCRIPTIVE_ROI_NAMES.values()}

    logger.info(f"Processing subjects: {subjects_to_process}")
    logger.info(f"Processing ROIs (using short names internally): {rois_to_process}")

    for subject_idx in subjects_to_process:
        logger.info(f"--- Processing Subject {subject_idx} ---")
        try:
            # Load full betas for the subject
            logger.info(f"Loading betas for subject {subject_idx}...")
            betas_xr = load_betas(
                subject=subject_idx,
                resolution=resolution,
                preprocessing=preprocessing,
                z_score=z_score,
                neuroid_filter=True
            )
            logger.info(f"Loaded betas shape: {betas_xr.shape} (presentations, neuroids)")
            logger.debug(f"Beta coordinates: {list(betas_xr.coords.keys())}")

            # Ensure neuroid dimension has coordinates for matching
            if not all(coord in betas_xr.coords for coord in ['x', 'y', 'z']):
                 logger.error(f"Betas for subject {subject_idx} missing x, y, or z coordinates for neuroids. Skipping.")
                 continue
            # Set multi-index for efficient selection based on coordinates
            betas_xr = betas_xr.set_index(neuroid=['x', 'y', 'z'])
            logger.debug(f"Betas shape after setting multi-index: {betas_xr.shape}")

            # Load ROI definitions for this subject
            logger.info(f"Loading ROI definitions for subject {subject_idx}...")
            rois_xr = load_rois(subject=subject_idx, resolution=resolution)
            logger.info(f"Loaded ROI definitions shape: {rois_xr.shape} (neuroids)")
            logger.debug(f"ROI definition coordinates: {list(rois_xr.coords.keys())}")

             # Ensure neuroid dimension has coordinates for matching
            if not all(coord in rois_xr.coords for coord in ['x', 'y', 'z']):
                 logger.error(f"ROI definitions for subject {subject_idx} missing x, y, or z coordinates for neuroids. Skipping.")
                 continue
            # Set multi-index for efficient selection based on coordinates
            rois_xr = rois_xr.set_index(neuroid=['x', 'y', 'z'])
            logger.debug(f"ROI definitions shape after setting multi-index: {rois_xr.shape}")

            subject_processed_any_roi = False # Flag to track if any ROI was processed for this subject
            for roi_name in rois_to_process:
                # Ensure the ROI exists in the main dictionary structure
                # Use the descriptive name for lookup in the final structure
                descriptive_roi_name = DESCRIPTIVE_ROI_NAMES.get(roi_name)
                if descriptive_roi_name is None:
                     logger.error(f"Internal error: ROI short name '{roi_name}' not found in DESCRIPTIVE_ROI_NAMES map. Skipping.")
                     continue
                if descriptive_roi_name not in all_processed_data:
                    # This case should ideally not happen due to initialization, but good practice
                    all_processed_data[descriptive_roi_name] = {}

                logger.info(f"Processing ROI '{roi_name}' (maps to '{descriptive_roi_name}') for Subject {subject_idx}...")
                try:
                    selectors = SPECIFIC_ROIS[roi_name]
                    logger.debug(f"Creating ROI selector for {roi_name} using selectors: {selectors}")
                    # Create the boolean mask based on the ROI definitions
                    roi_indices_mask = create_roi_selector(rois=rois_xr, selectors=selectors)
                    logger.debug(f"Generated ROI mask shape: {roi_indices_mask.shape}, Number of True values: {np.sum(roi_indices_mask)}")

                    # Get the neuroid multi-index (x, y, z) for the selected ROI voxels from the definition
                    target_neuroids_in_roi_def = rois_xr.isel(neuroid=roi_indices_mask).indexes['neuroid']
                    num_vox_in_def = len(target_neuroids_in_roi_def)
                    logger.debug(f"Found {num_vox_in_def} voxels in ROI definition for '{roi_name}'.")

                    if num_vox_in_def == 0:
                        logger.warning(f"ROI definition '{roi_name}' is empty for subject {subject_idx}. Skipping ROI.")
                        continue

                    # Find the intersection of voxels present in both the ROI definition and the loaded betas
                    beta_neuroids = betas_xr.indexes['neuroid']
                    common_neuroids = target_neuroids_in_roi_def.intersection(beta_neuroids)
                    num_common_vox = len(common_neuroids)
                    logger.info(f"Found {num_common_vox} common voxels between ROI def '{roi_name}' and loaded betas.")

                    if num_common_vox == 0:
                         logger.warning(f"ROI '{roi_name}' resulted in 0 overlapping voxels with loaded betas for subject {subject_idx}. Skipping ROI.")
                         continue

                    # Filter betas based on the common neuroids
                    roi_betas_xr = betas_xr.sel(neuroid=common_neuroids)
                    logger.info(f"Filtered betas shape for ROI '{roi_name}': {roi_betas_xr.shape} (presentations, neuroids)")

                    # Average across presentations, grouping by the 'stimulus' coordinate
                    logger.info(f"Averaging betas across presentations for ROI '{roi_name}' ({roi_betas_xr.sizes['neuroid']} voxels)...")
                    if "stimulus" not in roi_betas_xr.coords:
                        logger.error(f"Coordinate 'stimulus' not found in roi_betas_xr for ROI '{roi_name}', subject {subject_idx}. Cannot average. Skipping ROI.")
                        continue
                    avg_roi_betas_xr = roi_betas_xr.groupby("stimulus").mean("presentation")
                    logger.info(f"Averaged betas shape for ROI '{roi_name}': {avg_roi_betas_xr.shape} (stimuli, neuroids)")

                    # Convert to dictionary {stim_idx_str: numpy_array}
                    # Group again by 'stimulus' to iterate easily
                    stim_to_beta_map = {
                        # Convert stimulus index to string explicitly
                        # idx is already the stimulus ID (likely string or int convertible to string)
                        str(idx): data.values.astype(np.float32)
                        for idx, data in avg_roi_betas_xr.groupby("stimulus")
                    }
                    num_stim = len(stim_to_beta_map)
                    example_stim_id = next(iter(stim_to_beta_map)) if num_stim > 0 else None
                    example_shape = stim_to_beta_map[example_stim_id].shape if example_stim_id else "N/A"
                    logger.info(f"Finished averaging for ROI '{roi_name}'. Got data for {num_stim} stimuli. Shape per stimulus: {example_shape}")

                    # Store this subject's data for this ROI in the main dictionary using the DESCRIPTIVE name
                    all_processed_data[descriptive_roi_name][subject_idx] = stim_to_beta_map
                    subject_processed_any_roi = True

                except KeyError as e:
                    logger.error(f"KeyError processing ROI '{roi_name}' for subject {subject_idx}: {e}. This might indicate missing ROI labels or sources in the definitions. Skipping ROI for this subject.")
                except FileNotFoundError as e:
                     logger.error(f"Could not find underlying ROI definition files for subject {subject_idx}, resolution {resolution}: {e}. Skipping ROI for this subject.")
                except Exception as e:
                    logger.exception(f"Unexpected error processing ROI '{roi_name}' for subject {subject_idx}: {e}. Skipping ROI for this subject.")

            # Clean up memory for the large arrays before next subject
            del betas_xr, rois_xr
            gc.collect() # Explicitly collect garbage
            if subject_processed_any_roi:
                logger.info(f"Finished processing all requested ROIs for subject {subject_idx}.")
            else:
                logger.warning(f"Did not successfully process any ROIs for subject {subject_idx}.")

        except FileNotFoundError as e:
            logger.error(f"Failed to load base betas or ROI file for subject {subject_idx}: {e}. Skipping subject.")
        except Exception as e:
            logger.exception(f"Failed to process subject {subject_idx}: {e}")

    # --- Saving the combined processed data ---
    logger.info("--- Saving Combined Processed fMRI Data ---")

    # Check if any data was processed at all
    # Check based on descriptive names now
    if not any(all_processed_data[descriptive_name] for descriptive_name in all_processed_data):
        logger.warning("No fMRI data was successfully processed for any subject/ROI combination. Nothing to save.")
    else:
        # Construct the single output filename using the argument
        output_file = output_path / output_filename
        try:
            logger.info(f"Saving combined fMRI data for {len(subjects_to_process)} subject(s) and {len(all_processed_data)} ROI(s) to {output_file}...")
            # Use standard pickle dump
            with open(output_file, 'wb') as f:
                pickle.dump(all_processed_data, f)
            logger.info(f"Successfully saved combined fMRI data to {output_file}")
        except Exception as e:
            logger.exception(f"Failed to save combined fMRI data to {output_file}: {e}")

    logger.info("--- fMRI Data Processing Finished ---")


def process_stimuli_data(output_path: Path, output_filename: str, subject: int):
    """Loads, processes, and saves shared and unshared stimulus image data for a specific subject using StimulusSet."""
    logger.info(f"--- Starting Stimulus Data Processing for Subject {subject} ---")

    try:
        logger.info(f"Initializing StimulusSet for subject {subject}...")
        # Instantiate StimulusSet for the given subject
        stimulus_set = StimulusSet(subject=subject)
        # Total number of stimuli (shared + unshared for this subject)
        total_stimuli = len(stimulus_set)
        logger.info(f"StimulusSet initialized. Expecting {total_stimuli} total stimuli (shared + subject {subject}'s unshared).")

        # --- Pre-load data into NumPy arrays for efficiency ---
        logger.info("Loading stimuli data into NumPy arrays...")
        try:
            # Transpose to (stimulus, height, width, channel) for easier indexing
            shared_images_np = stimulus_set.stimuli_shared.transpose('stimulus', 'height', 'width', 'channel').values
            shared_stim_names = stimulus_set.stimuli_shared['stimulus'].values
            unshared_images_np = stimulus_set.stimuli_unshared.transpose('stimulus', 'height', 'width', 'channel').values
            unshared_stim_names = stimulus_set.stimuli_unshared['stimulus'].values
            logger.info("Successfully loaded data into NumPy arrays.")
        except Exception as e:
             logger.exception(f"Failed to load stimuli data into NumPy arrays: {e}")
             # If loading fails, exit or handle appropriately - here we raise to stop processing
             raise RuntimeError("Could not load stimuli data into memory.") from e
        # --- End Pre-loading ---

        # Convert to the desired dictionary format {stim_id_str: np.array (H, W, C)}
        logger.info("Converting stimuli to dictionary format {stim_id_str: np.array(H, W, C)}...")
        stimuli_dict = {}
        for i in range(total_stimuli):
            # Get data directly from NumPy arrays
            if i < N_STIMULI_SHARED:
                image_array = shared_images_np[i]
                stim_name = shared_stim_names[i]
            else:
                unshared_idx = i - N_STIMULI_SHARED
                image_array = unshared_images_np[unshared_idx]
                stim_name = unshared_stim_names[unshared_idx]

            # Use the stimulus name as the key (ensure string format)
            # stim_name from .values should already be appropriate type (often numpy string or object)
            stimuli_dict[str(stim_name)] = image_array

            if i % 1000 == 0: # Log progress periodically
                 logger.debug(f"Processed stimulus {i}/{total_stimuli} (name: {stim_name})")

        num_stim = len(stimuli_dict)
        example_stim_id = next(iter(stimuli_dict)) if num_stim > 0 else None
        example_shape = stimuli_dict[example_stim_id].shape if example_stim_id else "N/A"
        logger.info(f"Processed {num_stim} stimuli (shared + subject {subject}'s unshared). Example shape for stim '{example_stim_id}': {example_shape}")

        # --- Saving the processed stimulus data ---
        # Use the provided output_filename (which will include the subject)
        output_file = output_path / output_filename
        logger.info(f"Saving processed stimulus data for subject {subject} to {output_file}...")
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(stimuli_dict, f)
            logger.info(f"Successfully saved stimulus data to {output_file}")
        except Exception as e:
            logger.exception(f"Failed to save stimulus data to {output_file}: {e}")

    except FileNotFoundError as e:
        logger.error(f"Failed to load underlying stimuli files for subject {subject}: {e}. Ensure data is downloaded.")
    except Exception as e:
        logger.exception(f"Failed to process stimuli for subject {subject}: {e}")

    logger.info("--- Stimulus Data Processing Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Process NSD synthetic dataset - either fMRI responses or Stimuli.")
    parser.add_argument(
        '--data-type',
        type=str,
        required=True,
        choices=['fmri', 'stimuli'],
        help="Specify whether to process 'fmri' data or 'stimuli' images."
    )
    parser.add_argument(
        '--subject',
        type=int,
        # required=True, # Make subject required only if data-type is stimuli
        help="Specify the subject index (0-7) for stimuli processing (required if --data-type is stimuli)."
    )
    args = parser.parse_args()

    # Validate subject argument if processing stimuli
    if args.data_type == 'stimuli':
        if args.subject is None:
            parser.error("--subject is required when --data-type is stimuli")
        if not (0 <= args.subject <= 7):
             parser.error(f"--subject must be between 0 and 7, got {args.subject}")


    # --- Common Configuration ---
    output_dir = "datasets/neural/nsd_synthetic"
    # --------------------------

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.data_type == 'fmri':
        # Define output filename for fMRI
        output_filename = "fmri_responses.pkl"
        process_fmri_data(output_path, output_filename)
    elif args.data_type == 'stimuli':
        # Define output filename for stimuli, including the subject index
        output_filename = f"stimuli_subject_{args.subject}.pkl"
        process_stimuli_data(output_path, output_filename, args.subject)
    else:
        # This case should not be reachable due to argparse choices
        logger.error(f"Invalid data type specified: {args.data_type}")

    logger.info("--- Script Finished ---")

if __name__ == "__main__":
    main()
