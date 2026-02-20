"""Preprocess NSD fMRI data for all 8 subjects with shared/unique split.

Generates datasets/neural/nsd/nsd_data.pkl containing:
  - shared_ids: set of ~1000 nsdId ints (shared1000 stimuli seen by all subjects)
  - data: {region: {subject_idx: xr.DataArray}} for streams ROIs

Each DataArray has dims (stimulus, neuroid) with all ~10,000 stimuli per subject,
averaged across repetitions. Downstream loader splits train/test by shared vs unique.

Usage:
    python scripts/preprocess_data/preprocess_nsd.py
"""

import gc
import os
import pickle

import numpy as np
import xarray as xr

# Suppress bonner's verbose logging.
from loguru import logger

logger.remove()
logger.add(lambda _: None, level="WARNING")

# Must be set BEFORE importing bonner, which reads it at import time.
os.environ.setdefault("BONNER_DATASETS_HOME", "/data/shared/datasets")

# xarray >=2026 rejects drop_indexes on multi-index coords.
# bonner's load_betas calls drop_indexes("presentation") where presentation is a
# multi-index (session, trial). Fall back to reset_index when that happens.
_orig_drop_indexes = xr.DataArray.drop_indexes


def _safe_drop_indexes(self, coord_names, *, errors="raise"):
    try:
        return _orig_drop_indexes(self, coord_names, errors=errors)
    except ValueError:
        names = [coord_names] if isinstance(coord_names, str) else coord_names
        return self.reset_index(names)


xr.DataArray.drop_indexes = _safe_drop_indexes

from bonner.datasets.allen2021_natural_scenes import load_betas, load_rois
from bonner.datasets.allen2021_natural_scenes._stimuli import load_nsd_metadata

SUBJECTS = list(range(8))
REGIONS = {"early": ["early"], "ventral": ["ventral"]}
SAVE_PATH = "datasets/neural/nsd/nsd_data.pkl"


def filter_betas_by_roi(betas, rois, roi_labels, source="streams"):
    """Keep only voxels belonging to the given ROI labels from the specified source."""
    mask = np.zeros(rois.sizes["neuroid"], dtype=bool)
    for idx in rois.roi.values:
        roi_source, roi_label = idx[0], idx[1]
        if roi_label in roi_labels and roi_source == source:
            mask |= rois.sel(roi=idx).values > 0

    # Align betas and ROIs by (x, y, z) voxel coordinates
    betas = betas.set_xindex(["x", "y", "z"])
    rois = rois.set_xindex(["x", "y", "z"])
    roi_coords = set(rois.isel(neuroid=mask).indexes["neuroid"].to_list())
    beta_coords = set(betas.indexes["neuroid"].to_list())
    shared = list(roi_coords & beta_coords)
    return betas.sel(neuroid=shared)


def main():
    print("Processing NSD data (8 subjects, streams ROIs, shared1000 split)")

    metadata = load_nsd_metadata()
    shared_ids = set(int(x) for x in metadata.loc[metadata["shared1000"], "nsdId"])
    print(f"Shared1000 IDs: {len(shared_ids)}")

    data = {region: {} for region in REGIONS}

    for subj in SUBJECTS:
        print(f"\nSubject {subj}...")
        betas = load_betas(
            subject=subj,
            resolution="1pt8mm",
            preprocessing="fithrf_GLMdenoise_RR",
            z_score=True,
        )
        rois = load_rois(subject=subj, resolution="1pt8mm")

        for region, labels in REGIONS.items():
            roi_betas = filter_betas_by_roi(betas, rois, labels)
            averaged = roi_betas.groupby("stimulus").mean(dim="presentation")
            # .copy() breaks xarray reference chain to the full ~41 GB betas array
            data[region][subj] = averaged.copy()

            stim_ids = set(int(i) for i in averaged.coords["stimulus"].values)
            n_shared = len(stim_ids & shared_ids)
            print(
                f"  {region}: {roi_betas.sizes['neuroid']} voxels, "
                f"{averaged.sizes['stimulus']} stimuli ({n_shared} shared)"
            )
            del roi_betas, averaged

        del betas, rois
        gc.collect()

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    output = {"shared_ids": shared_ids, "data": data}
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(output, f)

    size_gb = sum(
        arr.values.nbytes for rd in data.values() for arr in rd.values()
    ) / (1024**3)
    print(f"\nSaved to {SAVE_PATH} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
