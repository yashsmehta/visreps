"""Preprocess NSD Synthetic fMRI data for all 8 subjects.

Generates:
  - datasets/neural/nsd_synthetic/nsd_synthetic_data_unfiltered.pkl: every ROI voxel
  - datasets/neural/nsd_synthetic/nsd_synthetic_data.pkl: only voxels whose
    subject-specific regular-NSD NCSNR exceeds the threshold (default 0.2, the
    same criterion and value used for regular NSD). This is the file the
    evaluation code loads.
  - datasets/neural/nsd_synthetic/stimuli/*.png  (220 shared stimulus images)

Pickle structure:
  {
      "data": {region_key: {subj_idx: xr.DataArray(stimulus, neuroid)}, ...},
      "shared_stimulus_names": [sorted list of 220 string names],
  }

Stimulus coordinate is a string name (e.g., "image_001"), not an integer nsdId.
Betas are averaged across repetitions per stimulus.

Usage:
    python scripts/preprocess_data/preprocess_nsd_synthetic.py                        # all regions
    python scripts/preprocess_data/preprocess_nsd_synthetic.py --regions V1 V2 hV4    # subset
    python scripts/preprocess_data/preprocess_nsd_synthetic.py --filter-only          # refilter
"""

import argparse
import gc
import os
import pickle

import numpy as np
import xarray as xr
from PIL import Image

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

from bonner.datasets.gifford2025_nsd_synthetic import load_betas
from bonner.datasets.gifford2025_nsd_synthetic._stimuli import (
    load_shared_stimuli,
    load_stimulus_information,
)
from bonner.datasets.allen2021_natural_scenes import load_rois, load_ncsnr
from preprocess_nsd import _ncsnr_lookup, _values_by_xyz

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
SAVE_DIR = "datasets/neural/nsd_synthetic"
SAVE_PATH = os.path.join(SAVE_DIR, "nsd_synthetic_data.pkl")
UNFILTERED_SAVE_PATH = os.path.join(SAVE_DIR, "nsd_synthetic_data_unfiltered.pkl")
DEFAULT_NCSNR_THRESHOLD = 0.2
STIMULI_DIR = os.path.join(SAVE_DIR, "stimuli")


def _build_roi_masks(rois, regions_to_extract):
    """Build boolean voxel masks for all requested regions in one pass."""
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
    """Average betas across repetitions per stimulus (string-keyed)."""
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


def filter_data_by_ncsnr(data, *, threshold=DEFAULT_NCSNR_THRESHOLD):
    """Keep voxels whose regular-NSD NCSNR exceeds ``threshold``.

    Reliability comes from the regular NSD scan, not from the synthetic one, so
    it is an independent criterion rather than a property of the test responses.
    """
    filtered = {region: {} for region in data}
    counts = []
    for subj in SUBJECTS:
        lookup = _ncsnr_lookup(load_ncsnr(
            subject=subj, resolution="1pt8mm", preprocessing="fithrf_GLMdenoise_RR"))
        for region, subject_data in data.items():
            responses = subject_data[subj]
            xyz = list(zip(*(responses.coords[c].values for c in "xyz")))
            ncsnr = _values_by_xyz(lookup, xyz)
            keep = np.flatnonzero(np.isfinite(ncsnr) & (ncsnr > threshold))
            if keep.size == 0:
                raise ValueError(f"No reliable voxels for subject {subj}, {region}")
            filtered[region][subj] = responses.isel(neuroid=keep).assign_coords(
                ncsnr=("neuroid", ncsnr[keep]))
            counts.append(dict(subject_idx=subj, region=region,
                               retained=int(keep.size), total=int(ncsnr.size)))
            print(f"  subj {subj} {region}: NCSNR > {threshold:g} keeps "
                  f"{keep.size}/{ncsnr.size} voxels")
        gc.collect()
    return filtered, counts


def _save(path, data, shared_stimulus_names, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(data=data, shared_stimulus_names=shared_stimulus_names, **extra), f)
    size_gb = sum(a.values.nbytes for rd in data.values() for a in rd.values()) / 1024 ** 3
    print(f"Saved {path} ({size_gb:.2f} GB, {len(data)} regions)")


def _save_shared_stimuli(shared_stimulus_names):
    """Save 220 shared stimulus images as PNGs (once, shared across subjects)."""
    os.makedirs(STIMULI_DIR, exist_ok=True)

    # Check if already saved
    existing = set(f.replace(".png", "") for f in os.listdir(STIMULI_DIR) if f.endswith(".png"))
    if existing >= set(shared_stimulus_names):
        print(f"  Stimuli already saved ({len(existing)} PNGs in {STIMULI_DIR})")
        return

    print(f"  Saving {len(shared_stimulus_names)} shared stimulus images...")
    images_xr = load_shared_stimuli()

    for name in shared_stimulus_names:
        out_path = os.path.join(STIMULI_DIR, f"{name}.png")
        if os.path.exists(out_path):
            continue
        img_data = images_xr.sel(stimulus=name).values  # (C, H, W) uint8
        img = Image.fromarray(np.transpose(img_data, (1, 2, 0)))
        img.save(out_path)

    print(f"  Saved {len(shared_stimulus_names)} PNGs to {STIMULI_DIR}")
    del images_xr


def main():
    parser = argparse.ArgumentParser(description="Preprocess NSD Synthetic fMRI data")
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
        help="Skip extraction; refilter the existing unfiltered archive.",
    )
    args = parser.parse_args()

    if args.filter_only:
        if not os.path.exists(UNFILTERED_SAVE_PATH):
            raise FileNotFoundError(
                f"{UNFILTERED_SAVE_PATH} does not exist; run a full extraction first")
        with open(UNFILTERED_SAVE_PATH, "rb") as f:
            unfiltered = pickle.load(f)
        filtered, counts = filter_data_by_ncsnr(
            unfiltered["data"], threshold=args.ncsnr_threshold)
        _save(SAVE_PATH, filtered, unfiltered["shared_stimulus_names"],
              ncsnr_threshold=args.ncsnr_threshold, voxel_counts=counts)
        return

    regions_to_extract = {r: REGIONS[r] for r in args.regions}
    print(f"Extracting regions: {list(regions_to_extract.keys())}")

    # Get shared stimulus names (first 220 of 284)
    stim_info = load_stimulus_information()
    shared_stimulus_names = sorted(stim_info["stimulus"].values[:220].tolist())
    shared_set = set(shared_stimulus_names)
    print(f"Shared stimuli: {len(shared_stimulus_names)}")

    # Save stimulus images
    _save_shared_stimuli(shared_stimulus_names)

    # Merge into existing pickle if present
    if os.path.exists(UNFILTERED_SAVE_PATH):
        print(f"Loading existing {UNFILTERED_SAVE_PATH} (merging, not overwriting)")
        with open(UNFILTERED_SAVE_PATH, "rb") as f:
            data = pickle.load(f)["data"]
    else:
        data = {}
    for region in regions_to_extract:
        data[region] = {}

    for subj in SUBJECTS:
        print(f"\nSubject {subj}...")
        betas = load_betas(
            subject=subj, resolution="1pt8mm",
            preprocessing="fithrf_GLMdenoise_RR", z_score=True,
        )
        rois = load_rois(subject=subj, resolution="1pt8mm")

        # Filter to shared stimuli only
        stim_mask = np.isin(betas.coords["stimulus"].values, list(shared_set))
        betas = betas.isel(presentation=stim_mask)
        print(f"  Filtered to {betas.sizes['presentation']} presentations of shared stimuli")

        # Coordinate lookup: (x,y,z) → beta index
        beta_xyz = list(zip(
            betas.coords["x"].values, betas.coords["y"].values, betas.coords["z"].values
        ))
        beta_coord_to_idx = {c: i for i, c in enumerate(beta_xyz)}
        roi_xyz = list(zip(
            rois.coords["x"].values, rois.coords["y"].values, rois.coords["z"].values
        ))

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

            n_stim = averaged.sizes["stimulus"]
            print(f"  {region}: {len(indices)} voxels, {n_stim} stimuli")
            del roi_betas, averaged

        del betas, rois
        gc.collect()

    print()
    _save(UNFILTERED_SAVE_PATH, data, shared_stimulus_names)
    filtered, counts = filter_data_by_ncsnr(data, threshold=args.ncsnr_threshold)
    _save(SAVE_PATH, filtered, shared_stimulus_names,
          ncsnr_threshold=args.ncsnr_threshold, voxel_counts=counts)


if __name__ == "__main__":
    main()
