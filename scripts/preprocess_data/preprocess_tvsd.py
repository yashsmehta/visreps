"""Preprocess TVSD (THINGS Ventral Stream Spiking Dataset) neural data.

Loads train and test splits from the bonner package:
  - Train: ~22,248 stimuli (each shown once)
  - Test:  100 stimuli (averaged across 30 repetitions)

Saves to datasets/neural/tvsd/fmri_responses.pkl with structure:
    data[region][subject_idx] = {"train": xr.DataArray, "test": xr.DataArray}
Electrode selection follows Papale et al. (2025, Neuron): keep electrodes whose
mean reliability across repetition pairs exceeds 0.3, the `reliab_th` used by the
authors' own `export_MUA.m` when preparing data for model training. The same
electrodes are used for train and test. All electrodes are saved to
fmri_responses_unfiltered.pkl.

Usage:
    python scripts/preprocess_data/preprocess_tvsd.py
    python scripts/preprocess_data/preprocess_tvsd.py --reliability-threshold 0.4
"""

import argparse
import os
import pickle

import numpy as np
from bonner.datasets.papale2025_tvsd import load_normalized_data

MONKEYS = {0: "F", 1: "N"}
SAVE_PATH = "datasets/neural/tvsd/fmri_responses.pkl"
UNFILTERED_SAVE_PATH = "datasets/neural/tvsd/fmri_responses_unfiltered.pkl"
# Papale et al. threshold on mean reliability (export_MUA.m: reliab_th = .3).
RELIABILITY_THRESHOLD = 0.3


def _zscore_repeats(responses):
    """Center and scale each (repetition, electrode) response profile over stimuli."""
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 3 or responses.shape[1] < 2:
        raise ValueError("Expected stimulus × repetition × electrode, with >=2 repeats")
    centered = responses - responses.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return centered / np.linalg.norm(centered, axis=0, keepdims=True)


def mean_reliability(responses):
    """Mean Pearson r over all repetition pairs; input: stimulus × repetition × electrode.

    Matches the `reliab` field of THINGS_normMUA.mat averaged across its
    C(n_reps, 2) columns — the quantity Papale et al. threshold at 0.3.
    """
    unit = _zscore_repeats(responses)
    n_repeats = unit.shape[1]
    with np.errstate(invalid="ignore"):
        pairwise = np.einsum("sre,sqe->rqe", unit, unit)
    upper = np.triu_indices(n_repeats, k=1)
    return pairwise[upper].mean(axis=0)


def oracle_correlation(responses):
    """Mean leave-one-repeat-out Pearson r; input: stimulus × repetition × electrode.

    Reproduces the `oracle` field of THINGS_normMUA.mat. Reported alongside the
    responses for reference; selection uses `mean_reliability`.
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 3 or responses.shape[1] < 2:
        raise ValueError("Expected stimulus × repetition × electrode, with >=2 repeats")
    others = (responses.sum(axis=1, keepdims=True) - responses) / (responses.shape[1] - 1)
    centered = responses - responses.mean(axis=0, keepdims=True)
    others -= others.mean(axis=0, keepdims=True)
    denom = np.sqrt((centered ** 2).sum(axis=0) * (others ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((centered * others).sum(axis=0) / denom).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reliability-threshold", type=float, default=RELIABILITY_THRESHOLD,
                        help="Keep electrodes with mean reliability strictly above this "
                             f"(default: {RELIABILITY_THRESHOLD}, as in Papale et al.).")
    args = parser.parse_args()
    if not np.isfinite(args.reliability_threshold) or not -1 <= args.reliability_threshold <= 1:
        parser.error("--reliability-threshold must be finite and between -1 and 1")
    data = {}
    unfiltered = {}

    for subj_idx, monkey in MONKEYS.items():
        print(f"Loading monkey {monkey} (subject {subj_idx})...")

        # Test: 100 stimuli × 30 reps → averaged
        raw_test = load_normalized_data(monkey=monkey, train=False)
        reps = raw_test.unstack("presentation").transpose("stimulus", "repetition", "neuroid").values
        reliability = mean_reliability(reps)
        oracle = oracle_correlation(reps)
        coords = {
            "neuroid": np.arange(reliability.size),
            "reliability": ("neuroid", reliability),
            "oracle": ("neuroid", oracle),
        }
        raw_test = raw_test.assign_coords(**coords)
        avg_test = raw_test.unstack("presentation").mean(dim="repetition")
        avg_test = avg_test.transpose("stimulus", "neuroid")

        # Train: ~22,248 stimuli (single presentation each)
        raw_train = load_normalized_data(monkey=monkey, train=True)
        raw_train = raw_train.assign_coords(**coords)
        train_data = raw_train.swap_dims({"presentation": "stimulus"})

        for region in np.unique(raw_test.coords["region"].values):
            test_mask = avg_test.coords["region"].values == region
            train_mask = train_data.coords["region"].values == region
            region_test = avg_test[:, test_mask].astype(np.float32)
            region_train = train_data[:, train_mask].astype(np.float32)

            unfiltered.setdefault(region, {})[subj_idx] = {
                "train": region_train,
                "test": region_test,
            }
            scores = region_test.coords["reliability"].values
            keep = np.flatnonzero(np.isfinite(scores) & (scores > args.reliability_threshold))
            if not keep.size:
                raise ValueError(f"No reliable electrodes for monkey {monkey}, {region}")
            data.setdefault(region, {})[subj_idx] = {
                split: responses.isel(neuroid=keep).assign_attrs(
                    reliability_threshold=args.reliability_threshold
                )
                for split, responses in unfiltered[region][subj_idx].items()
            }
            print(f"  {region}: reliability > {args.reliability_threshold:g} keeps "
                  f"{keep.size}/{scores.size} electrodes")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(UNFILTERED_SAVE_PATH, "wb") as f:
        pickle.dump(unfiltered, f)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
