"""Preprocess NSD fMRI data for all 8 subjects with shared/unique split.

Generates datasets/neural/nsd/nsd_data.pkl containing:
  - shared_ids: set of ~1000 nsdId ints (shared1000 stimuli seen by all subjects)
  - data: {region: {subject_idx: xr.DataArray}} for multiple ROI sources

Each DataArray has dims (stimulus, neuroid) with all ~10,000 stimuli per subject,
averaged across repetitions. Downstream loader splits train/test by shared vs unique.

Usage:
    python scripts/preprocess_data/preprocess_nsd.py                    # all regions
    python scripts/preprocess_data/preprocess_nsd.py --regions V1 V2 V3 hV4 FFA PPA
"""

import argparse
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
REGIONS = {
    "early": {"source": "streams", "labels": ["early"]},
    "ventral": {"source": "streams", "labels": ["ventral"]},
    "V1": {"source": "prf-visualrois", "labels": ["V1v", "V1d"]},
    "V2": {"source": "prf-visualrois", "labels": ["V2v", "V2d"]},
    "V3": {"source": "prf-visualrois", "labels": ["V3v", "V3d"]},
    "hV4": {"source": "prf-visualrois", "labels": ["hV4"]},
    "FFA": {"source": "floc-faces", "labels": ["FFA-1", "FFA-2"]},
    "PPA": {"source": "floc-places", "labels": ["PPA"]},
}
SAVE_PATH = "datasets/neural/nsd/nsd_data.pkl"


def _build_roi_masks(rois, regions_to_extract):
    """Build boolean voxel masks for all requested regions in one pass."""
    # Invert: (source, label) → [region_keys] for O(1) lookup
    needed = {}
    for region, rcfg in regions_to_extract.items():
        for label in rcfg["labels"]:
            needed.setdefault((rcfg["source"], label), []).append(region)

    masks = {r: np.zeros(rois.sizes["neuroid"], dtype=bool) for r in regions_to_extract}
    for idx in rois.roi.values:
        key = (idx[0], idx[1])
        if key in needed:
            vals = rois.sel(roi=idx).values > 0
            for region in needed[key]:
                masks[region] |= vals

    return masks


def _average_by_stimulus(roi_betas):
    """Average betas across repetitions per stimulus (numpy, faster than xarray groupby)."""
    stim_labels = roi_betas.coords["stimulus"].values
    unique_stim, inverse = np.unique(stim_labels, return_inverse=True)
    data = roi_betas.values
    sums = np.zeros((len(unique_stim), data.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, data)
    counts = np.bincount(inverse, minlength=len(unique_stim))
    averaged = (sums / counts[:, None]).astype(data.dtype)
    return xr.DataArray(averaged, dims=("stimulus", "neuroid"), coords={"stimulus": unique_stim})


def main():
    parser = argparse.ArgumentParser(description="Preprocess NSD fMRI data")
    parser.add_argument(
        "--regions", nargs="+", default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()), metavar="REGION",
        help=f"Regions to extract. Choices: {list(REGIONS.keys())}. Default: all.",
    )
    args = parser.parse_args()
    regions_to_extract = {r: REGIONS[r] for r in args.regions}
    print(f"Extracting regions: {list(regions_to_extract.keys())}")

    metadata = load_nsd_metadata()
    shared_ids = set(int(x) for x in metadata.loc[metadata["shared1000"], "nsdId"])
    print(f"Shared1000 IDs: {len(shared_ids)}")

    # Merge into existing pickle: only requested regions are overwritten, rest preserved
    if os.path.exists(SAVE_PATH):
        print(f"Loading existing {SAVE_PATH} (merging, not overwriting)")
        with open(SAVE_PATH, "rb") as f:
            data = pickle.load(f)["data"]
    else:
        data = {}
    for region in regions_to_extract:
        data[region] = {}  # reset only regions being extracted

    for subj in SUBJECTS:
        print(f"\nSubject {subj}...")
        betas = load_betas(
            subject=subj, resolution="1pt8mm",
            preprocessing="fithrf_GLMdenoise_RR", z_score=True,
        )
        rois = load_rois(subject=subj, resolution="1pt8mm")

        # Coordinate lookup once per subject: (x,y,z) → beta index
        beta_xyz = list(zip(betas.coords["x"].values, betas.coords["y"].values, betas.coords["z"].values))
        beta_coord_to_idx = {c: i for i, c in enumerate(beta_xyz)}
        roi_xyz = list(zip(rois.coords["x"].values, rois.coords["y"].values, rois.coords["z"].values))

        roi_masks = _build_roi_masks(rois, regions_to_extract)

        for region in regions_to_extract:
            indices = sorted(
                beta_coord_to_idx[roi_xyz[i]]
                for i in np.where(roi_masks[region])[0]
                if roi_xyz[i] in beta_coord_to_idx
            )
            roi_betas = betas.isel(neuroid=indices)
            averaged = _average_by_stimulus(roi_betas)
            data[region][subj] = averaged

            n_shared = len(set(int(i) for i in averaged.coords["stimulus"].values) & shared_ids)
            print(f"  {region}: {len(indices)} voxels, {averaged.sizes['stimulus']} stimuli ({n_shared} shared)")
            del roi_betas, averaged

        del betas, rois
        gc.collect()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"shared_ids": shared_ids, "data": data}, f)

    size_gb = sum(arr.values.nbytes for rd in data.values() for arr in rd.values()) / (1024**3)
    print(f"\nSaved to {SAVE_PATH} ({size_gb:.2f} GB, {len(data)} regions)")


if __name__ == "__main__":
    main()
