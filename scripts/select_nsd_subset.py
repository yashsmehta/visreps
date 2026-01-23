"""Process and save NSD fMRI data for specified subjects and ROIs."""

import os
import pickle
import numpy as np
from loguru import logger

logger.remove()
logger.add(lambda _: None, level="WARNING")

from bonner.datasets.allen2021_natural_scenes import load_rois, load_betas

# =============================================================================
# Configuration
# =============================================================================
roi_type = "streams"  # "streams" (early/ventral) or "finegrained" (V1-V4, FFA, etc.)
subjects = [0, 1, 3]
output_dir = f"/home/ymehta3/research/VisionAI/visreps/datasets/neural/nsd/{roi_type}"

# =============================================================================
# ROI definitions
# =============================================================================
STREAMS_ROIS = {
    "early": ["early visual stream"],
    "ventral": ["ventral visual stream"],
}

FINEGRAINED_ROIS = {
    "V1": ["V1v", "V1d"],
    "V2": ["V2v", "V2d"],
    "V3": ["V3v", "V3d"],
    "V4": ["hV4"],
    "VO": ["VO1", "VO2"],
    "PIT": ["PIT"],
    "OFA": ["OFA"],
    "FFA": ["FFA-1", "FFA-2"],
    "FBA": ["FBA-1", "FBA-2"],
    "OPA": ["OPA"],
    "PPA": ["PPA"],
    "TE2p": ["TE2p"],
}


def filter_betas_by_roi(betas, rois, roi_labels, source=None):
    """Filter betas to only include voxels in specified ROI labels.
    
    Args:
        betas: Neural response data
        rois: ROI masks
        roi_labels: List of ROI label strings to include
        source: If specified, only use ROIs from this source (e.g., "streams", "Kastner2015")
    """
    # Index ROIs by label
    rois_by_label = {}
    for idx in rois.roi.values:
        label = idx[1]
        if label not in rois_by_label:
            rois_by_label[label] = []
        rois_by_label[label].append(idx)

    # Create combined boolean mask on neuroid dimension
    neuroid_mask = np.zeros(rois.sizes["neuroid"], dtype=bool)

    for roi_label in roi_labels:
        if roi_label not in rois_by_label:
            continue
        entries = rois_by_label[roi_label]
        # Filter by source if specified, otherwise prefer Kastner2015 for fine-grained
        if source:
            use_entries = [idx for idx in entries if idx[0] == source] or entries
        else:
            use_entries = [idx for idx in entries if idx[0] == "Kastner2015"] or entries

        for idx in use_entries:
            roi_data = rois.sel(roi=idx).values > 0
            neuroid_mask |= roi_data

    # Set multi-index on both to enable proper selection
    betas = betas.set_xindex(["x", "y", "z"])
    rois = rois.set_xindex(["x", "y", "z"])

    # Get ROI neuroid multi-index values where mask is True
    roi_neuroids = rois.isel(neuroid=neuroid_mask).indexes["neuroid"]

    # Get intersection with beta neuroids
    valid_neuroids = list(set(roi_neuroids.to_list()) & set(betas.indexes["neuroid"].to_list()))

    return betas.sel(neuroid=valid_neuroids)

def main():
    # Select ROI config based on type
    if roi_type == "streams":
        roi_regions = STREAMS_ROIS
        source = "streams"
        output_subdir = "nsd_streams"
    else:
        roi_regions = FINEGRAINED_ROIS
        source = None  # Will default to Kastner2015
        output_subdir = "nsd_full"

    output_path = os.path.join(output_dir, output_subdir)

    print("=" * 80)
    print(f"PROCESSING NSD DATA ({roi_type} ROIs)")
    print("=" * 80)
    print(f"Regions: {list(roi_regions.keys())}")

    fmri_data = {region: {} for region in roi_regions}

    for subject_idx in subjects:
        print(f"\nProcessing Subject {subject_idx}...")
        betas = load_betas(subject=subject_idx, resolution="1pt8mm", 
                          preprocessing="fithrf_GLMdenoise_RR", z_score=True)
        rois = load_rois(subject=subject_idx, resolution="1pt8mm")

        for region_name, roi_labels in roi_regions.items():
            print(f"  {region_name}...", end=" ")
            roi_betas = filter_betas_by_roi(betas, rois, roi_labels, source=source)
            averaged_betas = roi_betas.groupby("stimulus").mean(dim="presentation")
            fmri_data[region_name][subject_idx] = averaged_betas
            print(f"{roi_betas.sizes['neuroid']} voxels, {averaged_betas.sizes['stimulus']} stimuli")

    # Calculate file size
    print("\n" + "=" * 80)
    print("FILE SIZE ESTIMATE")
    print("=" * 80)

    fmri_size_gb = sum(
        xr_data.values.nbytes for subjects_data in fmri_data.values() 
        for xr_data in subjects_data.values()
    ) / (1024**3)
    print(f"\nfmri_responses.pkl: {fmri_size_gb:.2f} GB")

    # Save data
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)

    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "fmri_responses.pkl")
    print(f"\nSaving to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(fmri_data, f)

    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
