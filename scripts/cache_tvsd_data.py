"""
Cache TVSD (THINGS Ventral Stream Spiking Dataset) neural data as a pickle file.

Loads data from the bonner package, averages across 30 repetitions,
and saves in the same structure as NSD: data[region][subject_idx] → xr.DataArray.

Usage:
    python scripts/cache_tvsd_data.py
"""

import os
import pickle
import numpy as np
import xarray as xr
from bonner.datasets.papale2025_tvsd import load_normalized_data

MONKEYS = {0: "F", 1: "N"}
SAVE_DIR = "datasets/neural/tvsd"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    data = {}  # data[region][subject_idx] → xr.DataArray (stimulus, neuroid)

    for subj_idx, monkey in MONKEYS.items():
        print(f"Loading monkey {monkey} (subject_idx={subj_idx})...")
        raw = load_normalized_data(monkey=monkey, train=False)

        # Unstack and average across 30 repetitions
        averaged = raw.unstack("presentation").mean(dim="repetition")
        # averaged shape: (neuroid, stimulus) → transpose to (stimulus, neuroid)
        averaged = averaged.transpose("stimulus", "neuroid")

        regions = np.unique(raw.coords["region"].values)
        for region in regions:
            region_mask = averaged.coords["region"].values == region
            region_data = averaged[:, region_mask]
            print(f"  {region}: {region_data.shape} (stimulus, neuroid)")

            if region not in data:
                data[region] = {}
            data[region][subj_idx] = region_data.astype(np.float32)

    save_path = os.path.join(SAVE_DIR, "fmri_responses.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {save_path}")

    # Verify
    with open(save_path, "rb") as f:
        check = pickle.load(f)
    print(f"Regions: {list(check.keys())}")
    for region in check:
        print(f"  {region}: subjects {list(check[region].keys())}")
        for subj, arr in check[region].items():
            print(f"    subject {subj}: {arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
