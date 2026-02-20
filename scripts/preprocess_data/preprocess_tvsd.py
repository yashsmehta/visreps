"""Preprocess TVSD (THINGS Ventral Stream Spiking Dataset) neural data.

Loads train and test splits from the bonner package:
  - Train: ~22,248 stimuli (each shown once)
  - Test:  100 stimuli (averaged across 30 repetitions)

Saves to datasets/neural/tvsd/fmri_responses.pkl with structure:
    data[region][subject_idx] = {"train": xr.DataArray, "test": xr.DataArray}

Usage:
    python scripts/preprocess_data/preprocess_tvsd.py
"""

import os
import pickle

import numpy as np
from bonner.datasets.papale2025_tvsd import load_normalized_data

MONKEYS = {0: "F", 1: "N"}
SAVE_PATH = "datasets/neural/tvsd/fmri_responses.pkl"


def main():
    data = {}

    for subj_idx, monkey in MONKEYS.items():
        print(f"Loading monkey {monkey} (subject {subj_idx})...")

        # Test: 100 stimuli × 30 reps → averaged
        raw_test = load_normalized_data(monkey=monkey, train=False)
        avg_test = raw_test.unstack("presentation").mean(dim="repetition")
        avg_test = avg_test.transpose("stimulus", "neuroid")

        # Train: ~22,248 stimuli (single presentation each)
        raw_train = load_normalized_data(monkey=monkey, train=True)
        train_data = raw_train.swap_dims({"presentation": "stimulus"})

        for region in np.unique(raw_test.coords["region"].values):
            test_mask = avg_test.coords["region"].values == region
            train_mask = train_data.coords["region"].values == region
            region_test = avg_test[:, test_mask].astype(np.float32)
            region_train = train_data[:, train_mask].astype(np.float32)

            data.setdefault(region, {})[subj_idx] = {
                "train": region_train,
                "test": region_test,
            }
            print(f"  {region}: train {region_train.shape}, test {region_test.shape}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
