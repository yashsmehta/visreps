"""Preprocess NSD fMRI data for all 8 subjects with shared/unique split.

Generates two files:
  - nsd_data_unfiltered.pkl: every voxel in each requested ROI
  - nsd_data.pkl: only voxels whose subject-specific NCSNR exceeds the threshold

Both contain:
  - shared_ids: set of ~1000 nsdId ints (shared1000 stimuli seen by all subjects)
  - data: {region: {subject_idx: xr.DataArray}} for multiple ROI sources

Each DataArray has dims (stimulus, neuroid) with all ~10,000 stimuli per subject,
averaged across repetitions. Downstream loader splits train/test by shared vs unique.

Usage:
    python scripts/preprocess_data/preprocess_nsd.py                    # all regions
    python scripts/preprocess_data/preprocess_nsd.py --regions V1 V2 V3 hV4 FFA PPA
    python scripts/preprocess_data/preprocess_nsd.py --filter-only     # rebuild filtered file

The default NCSNR threshold is 0.2. NCSNR is one reliability estimate per
subject and voxel; it is not the number of stimulus repetitions.
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

from bonner.datasets.allen2021_natural_scenes import (
    load_betas,
    load_brain_mask,
    load_ncsnr,
    load_rois,
    load_validity,
)
from bonner.datasets.allen2021_natural_scenes._data import N_SESSIONS
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
UNFILTERED_SAVE_PATH = "datasets/neural/nsd/nsd_data_unfiltered.pkl"
DEFAULT_NCSNR_THRESHOLD = 0.2


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
    coords = {"stimulus": unique_stim}
    for name in ("neuroid", "x", "y", "z"):
        if name in roi_betas.coords:
            coords[name] = roi_betas.coords[name]
    return xr.DataArray(averaged, dims=("stimulus", "neuroid"), coords=coords)


def _ncsnr_lookup(volume):
    """Index a flattened NSD NCSNR volume by ``(x, y, z)``."""
    volume_xyz = zip(
        volume.coords["x"].values,
        volume.coords["y"].values,
        volume.coords["z"].values,
    )
    return {coord: value for coord, value in zip(volume_xyz, volume.values)}


def _values_by_xyz(lookup, xyz):
    """Return values from an indexed NSD volume in ``xyz`` order."""
    missing = [coord for coord in xyz if coord not in lookup]
    if missing:
        raise ValueError(f"NCSNR volume is missing {len(missing)} beta coordinates")
    return np.asarray([lookup[coord] for coord in xyz])


def _restore_legacy_roi_coordinates(data):
    """Restore coordinates omitted by older versions of this preprocessor."""
    missing_regions = {
        region for region, subject_data in data.items()
        if any("x" not in responses.coords for responses in subject_data.values())
    }
    if not missing_regions:
        return

    region_configs = {region: REGIONS[region] for region in missing_regions}
    print(f"Restoring legacy voxel coordinates for: {sorted(missing_regions)}")
    for subj in SUBJECTS:
        rois = load_rois(subject=subj, resolution="1pt8mm")
        roi_masks = _build_roi_masks(rois, region_configs)
        brain = load_brain_mask(subject=subj, resolution="1pt8mm").values.ravel()
        validity = load_validity(subject=subj, resolution="1pt8mm")
        valid = validity.isel(session=np.arange(N_SESSIONS[subj])).all("session").values.ravel()
        beta_valid = brain.astype(bool) & valid.astype(bool)
        roi_xyz = list(zip(
            rois.coords["x"].values,
            rois.coords["y"].values,
            rois.coords["z"].values,
        ))

        for region in missing_regions:
            responses = data[region][subj]
            selected_xyz = [
                coord for coord, in_roi, is_valid
                in zip(roi_xyz, roi_masks[region], beta_valid)
                if in_roi and is_valid
            ]
            if len(selected_xyz) != responses.sizes["neuroid"]:
                raise ValueError(
                    f"Cannot restore {region} subject {subj}: reconstructed "
                    f"{len(selected_xyz)} coordinates for {responses.sizes['neuroid']} voxels"
                )
            x, y, z = map(np.asarray, zip(*selected_xyz))
            data[region][subj] = responses.assign_coords(
                x=("neuroid", x), y=("neuroid", y), z=("neuroid", z)
            )


def filter_data_by_ncsnr(data, *, threshold=DEFAULT_NCSNR_THRESHOLD):
    """Filter preprocessed ROI arrays using subject-specific NSD NCSNR."""
    _restore_legacy_roi_coordinates(data)
    filtered = {region: {} for region in data}
    counts = {region: {} for region in data}
    ncsnr_cache = {}

    for region, subject_data in data.items():
        for subj, responses in subject_data.items():
            if subj not in ncsnr_cache:
                ncsnr_cache[subj] = _ncsnr_lookup(
                    load_ncsnr(
                        subject=subj,
                        resolution="1pt8mm",
                        preprocessing="fithrf_GLMdenoise_RR",
                    )
                )
            xyz = list(zip(
                responses.coords["x"].values,
                responses.coords["y"].values,
                responses.coords["z"].values,
            ))
            roi_ncsnr = _values_by_xyz(ncsnr_cache[subj], xyz)
            keep = np.isfinite(roi_ncsnr) & (roi_ncsnr > threshold)
            selected = responses.isel(neuroid=np.flatnonzero(keep)).assign_coords(
                ncsnr=("neuroid", roi_ncsnr[keep])
            )
            filtered[region][subj] = selected
            counts[region][subj] = (int(keep.sum()), int(keep.size))

    return filtered, counts


def main():
    parser = argparse.ArgumentParser(description="Preprocess NSD fMRI data")
    parser.add_argument(
        "--regions", nargs="+", default=list(REGIONS.keys()),
        choices=list(REGIONS.keys()), metavar="REGION",
        help=f"Regions to extract. Choices: {list(REGIONS.keys())}. Default: all.",
    )
    parser.add_argument(
        "--ncsnr-threshold", type=float, default=DEFAULT_NCSNR_THRESHOLD,
        help="Keep voxels with NCSNR strictly above this value (default: 0.2).",
    )
    parser.add_argument(
        "--filter-only", action="store_true",
        help=f"Rebuild {SAVE_PATH} from {UNFILTERED_SAVE_PATH} without reloading betas.",
    )
    args = parser.parse_args()

    if args.filter_only:
        if not os.path.exists(UNFILTERED_SAVE_PATH):
            raise FileNotFoundError(
                f"{UNFILTERED_SAVE_PATH} does not exist; preserve the unfiltered data there first"
            )
        print(f"Loading {UNFILTERED_SAVE_PATH}")
        with open(UNFILTERED_SAVE_PATH, "rb") as f:
            unfiltered = pickle.load(f)
        filtered_data, counts = filter_data_by_ncsnr(
            unfiltered["data"], threshold=args.ncsnr_threshold,
        )
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                "shared_ids": unfiltered["shared_ids"],
                "data": filtered_data,
                "ncsnr_threshold": args.ncsnr_threshold,
                "voxel_counts": counts,
            }, f)
        print(f"Saved NCSNR > {args.ncsnr_threshold:g} data to {SAVE_PATH}")
        for region, subject_counts in counts.items():
            summary = ", ".join(
                f"subj{subj + 1:02}: {kept}/{total}"
                for subj, (kept, total) in subject_counts.items()
            )
            print(f"  {region}: {summary}")
        return

    regions_to_extract = {r: REGIONS[r] for r in args.regions}
    print(f"Extracting regions: {list(regions_to_extract.keys())}")

    metadata = load_nsd_metadata()
    shared_ids = set(int(x) for x in metadata.loc[metadata["shared1000"], "nsdId"])
    print(f"Shared1000 IDs: {len(shared_ids)}")

    # Merge into the unfiltered source: only requested regions are overwritten.
    source_path = UNFILTERED_SAVE_PATH if os.path.exists(UNFILTERED_SAVE_PATH) else SAVE_PATH
    if os.path.exists(source_path):
        print(f"Loading existing {source_path} (merging, not overwriting)")
        with open(source_path, "rb") as f:
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

    filtered_data, counts = filter_data_by_ncsnr(
        data, threshold=args.ncsnr_threshold,
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(UNFILTERED_SAVE_PATH, "wb") as f:
        pickle.dump({"shared_ids": shared_ids, "data": data}, f)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({
            "shared_ids": shared_ids,
            "data": filtered_data,
            "ncsnr_threshold": args.ncsnr_threshold,
            "voxel_counts": counts,
        }, f)

    size_gb = sum(arr.values.nbytes for rd in data.values() for arr in rd.values()) / (1024**3)
    print(f"\nSaved unfiltered data to {UNFILTERED_SAVE_PATH} ({size_gb:.2f} GB)")
    print(f"Saved NCSNR > {args.ncsnr_threshold:g} data to {SAVE_PATH}")
    for region, subject_counts in counts.items():
        summary = ", ".join(
            f"subj{subj + 1:02}: {kept}/{total}"
            for subj, (kept, total) in subject_counts.items()
        )
        print(f"  {region}: {summary}")


if __name__ == "__main__":
    main()
