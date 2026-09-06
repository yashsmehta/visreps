"""Extract per-repetition NSD betas for the shared1000 stimuli.

``preprocess_nsd.py`` averages each stimulus over its 3 presentations before
saving, which is what evaluation needs but leaves no way to measure how
reliable a subject's RDM is. This script makes a second, much smaller file
holding the *unaveraged* trials for the shared (test) stimuli only, which
``visreps.analysis.noise_ceiling`` uses for the within-subject split-half
ceiling.

Voxels are matched by (x, y, z) against ``nsd_data.pkl`` so the voxel set and
its order are identical to what evaluations score on -- no re-derivation of the
NCSNR filter, so the two files cannot drift apart.

Betas are read straight from the session HDF5s rather than through
``bonner.load_betas``. Those files are chunked one voxel per chunk, so pulling
the few thousand ROI voxels costs ~0.1 s per session against ~130 s to
transpose and z-score the whole 1 GB volume. z-scoring is per voxel within a
session, so restricting voxels first is exactly equivalent -- and ``--verify``
proves it by re-averaging the trials and diffing against ``nsd_data.pkl``.

Trials are stored flat rather than padded, since subjects who did not complete
all 40 sessions have some shared stimuli with fewer than 3 presentations:

    data[region][subj] = {
        "stimulus_ids":   (n_stimuli,) int   -- nsdId, ascending
        "trial_stimulus": (n_trials,)  int   -- index into stimulus_ids
        "betas":          (n_trials, n_voxels) float32
    }

Usage:
    python scripts/preprocess_data/preprocess_nsd_repetitions.py
    python scripts/preprocess_data/preprocess_nsd_repetitions.py --regions early ventral
    python scripts/preprocess_data/preprocess_nsd_repetitions.py --no-verify
"""

import argparse
import gc
import os
import pickle
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# preprocess_nsd sets BONNER_DATASETS_HOME and silences loguru at import time;
# importing it first keeps that setup in one place.
from preprocess_nsd import REGIONS, SUBJECTS  # noqa: E402

from bonner.datasets.allen2021_natural_scenes._data import (  # noqa: E402
    CACHE_PATH,
    N_SESSIONS,
    N_TRIALS_PER_SESSION,
)
from bonner.datasets.allen2021_natural_scenes._stimuli import load_nsd_metadata  # noqa: E402

FILTERED_PATH = "datasets/neural/nsd/nsd_data.pkl"
SAVE_PATH = "datasets/neural/nsd/nsd_repetitions.pkl"
PREPROCESSING = "fithrf_GLMdenoise_RR"
RESOLUTION = "1pt8mm"


def session_path(subject, session):
    return (
        CACHE_PATH / "nsddata_betas" / "ppdata" / f"subj{subject + 1:02}"
        / f"func{RESOLUTION}" / f"betas_{PREPROCESSING}"
        / f"betas_session{session + 1:02}.hdf5"
    )


def trial_stimulus_ids(metadata, subject):
    """nsdId shown on each trial, in (session, trial) order -- bonner's ordering."""
    cols = [f"subject{subject}_rep{k}" for k in range(3)]
    reps = metadata.loc[:, cols].to_numpy()
    nsd_ids, _ = np.nonzero(reps)
    trials = reps[np.nonzero(reps)] - 1  # stored 1-indexed

    grid = np.empty((max(N_SESSIONS), N_TRIALS_PER_SESSION), dtype=np.int64)
    grid[trials // N_TRIALS_PER_SESSION, trials % N_TRIALS_PER_SESSION] = nsd_ids
    return grid[: N_SESSIONS[subject]].ravel()


def region_voxels(filtered, region, subj):
    """(x, y, z) of the evaluated voxels for one region/subject, in stored order."""
    arr = filtered["data"][region][subj]
    missing = [c for c in ("x", "y", "z") if c not in arr.coords]
    if missing:
        raise ValueError(
            f"{FILTERED_PATH} lacks {missing} coords for {region}/subj{subj}; "
            "rebuild it with preprocess_nsd.py --filter-only first"
        )
    return list(zip(
        arr.coords["x"].values, arr.coords["y"].values, arr.coords["z"].values
    ))


def read_subject(subject, voxels, keep_trials):
    """Z-scored betas for `voxels` on `keep_trials`, read straight from the HDF5s.

    Args:
        voxels: list of (x, y, z), the union across regions.
        keep_trials: boolean mask over all of the subject's trials.

    Returns:
        (n_kept_trials, n_voxels) float32.
    """
    x = np.fromiter((v[0] for v in voxels), dtype=np.int64, count=len(voxels))
    y = np.fromiter((v[1] for v in voxels), dtype=np.int64, count=len(voxels))
    z = np.fromiter((v[2] for v in voxels), dtype=np.int64, count=len(voxels))

    chunks = []
    for session in range(N_SESSIONS[subject]):
        lo = session * N_TRIALS_PER_SESSION
        wanted = keep_trials[lo : lo + N_TRIALS_PER_SESSION]

        with h5py.File(session_path(subject, session), "r") as f:
            ds = f["betas"]  # (presentation, z, y, x), one chunk per voxel
            block = np.empty((ds.shape[0], len(voxels)), dtype=np.float32)
            for k in range(len(voxels)):
                block[:, k] = ds[:, z[k], y[k], x[k]]

        # Per-voxel z-score within the session, exactly as bonner.load_betas does.
        block -= block.mean(axis=0)
        block /= block.std(axis=0)
        chunks.append(block[wanted])
        del block

    return np.concatenate(chunks, axis=0)


def verify(data, filtered, regions, tol=1e-4):
    """Re-average the extracted trials and diff against nsd_data.pkl."""
    print("\nVerifying against the averaged file...")
    worst = 0.0
    for region in regions:
        for subj, entry in data[region].items():
            stim_ids = entry["stimulus_ids"]
            means = np.zeros((len(stim_ids), entry["betas"].shape[1]), dtype=np.float64)
            np.add.at(means, entry["trial_stimulus"], entry["betas"])
            means /= np.bincount(
                entry["trial_stimulus"], minlength=len(stim_ids))[:, None]

            reference = filtered["data"][region][subj].sel(
                stimulus=stim_ids.tolist()).values
            diff = float(np.abs(means - reference).max())
            worst = max(worst, diff)
            if diff > tol:
                raise AssertionError(
                    f"{region}/subj{subj}: max |diff| {diff:.3e} exceeds {tol:g}"
                )
    print(f"  OK -- max |diff| vs nsd_data.pkl across all region/subject pairs: {worst:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions", nargs="+", default=list(REGIONS.keys()),
                   choices=list(REGIONS.keys()), metavar="REGION")
    p.add_argument("--filtered", default=FILTERED_PATH)
    p.add_argument("--out", default=SAVE_PATH)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    print(f"Loading evaluated voxel set from {args.filtered}")
    with open(args.filtered, "rb") as f:
        filtered = pickle.load(f)

    regions = [r for r in args.regions if r in filtered["data"]]
    skipped = sorted(set(args.regions) - set(regions))
    if skipped:
        print(f"  skipping {skipped}: not present in {args.filtered}")

    metadata = load_nsd_metadata()
    shared_ids = set(int(x) for x in metadata.loc[metadata["shared1000"], "nsdId"])
    print(f"Shared1000 IDs: {len(shared_ids)}")

    data = {region: {} for region in regions}

    for subj in SUBJECTS:
        print(f"\nSubject {subj}...", flush=True)

        # One read covers every region: take the union, slice it up afterwards.
        per_region = {r: region_voxels(filtered, r, subj) for r in regions}
        union = sorted({v for vs in per_region.values() for v in vs})
        position = {v: i for i, v in enumerate(union)}

        trial_stim = trial_stimulus_ids(metadata, subj)
        keep = np.isin(trial_stim, list(shared_ids))
        stimulus_ids = np.unique(trial_stim[keep])
        trial_index = np.searchsorted(stimulus_ids, trial_stim[keep])
        counts = np.bincount(trial_index, minlength=len(stimulus_ids))
        print(f"  {int(keep.sum())} shared trials over {len(stimulus_ids)} stimuli "
              f"(reps: min {counts.min()}, median {int(np.median(counts))}, "
              f"max {counts.max()}); {len(union)} union voxels", flush=True)

        betas = read_subject(subj, union, keep)

        for region in regions:
            cols = [position[v] for v in per_region[region]]
            data[region][subj] = {
                "stimulus_ids": stimulus_ids.astype(np.int64),
                "trial_stimulus": trial_index.astype(np.int32),
                "betas": np.ascontiguousarray(betas[:, cols]),
            }
            print(f"    {region}: {len(cols)} voxels x {betas.shape[0]} trials")

        del betas
        gc.collect()

    if not args.no_verify:
        verify(data, filtered, regions)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"shared_ids": shared_ids, "data": data}, f)

    size_gb = sum(
        d["betas"].nbytes for rd in data.values() for d in rd.values()
    ) / (1024 ** 3)
    print(f"\nSaved -> {args.out} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
