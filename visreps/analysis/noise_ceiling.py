"""Within-subject RSA noise ceiling for NSD.

The ceiling is the highest score any model could reach: a subject's measured
RDM carries trial noise that nothing can predict. It is estimated per subject
from the repeated presentations of each stimulus, then averaged over subjects.

    1. split each stimulus's trials 1-vs-1, build an RDM from each half and
       correlate them  ->  reliability of a *single-trial* RDM
    2. Spearman-Brown up to the number of trials actually averaged (~3)
       ->  reliability of the RDM that models are scored against
    3. sqrt()  ->  a noiseless model correlates with noisy data at
       sqrt(reliability), which is the ceiling

RDMs are built and compared with the same functions that score models, so the
ceiling is on the scale of the numbers in results.db.

``nsd_ceiling`` needs scripts/preprocess_data/preprocess_nsd_repetitions.py to
have run; ``tvsd_ceiling`` reads its repetitions straight from the bonner
package, which is small enough not to need a cached file.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

import visreps.utils as utils
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
from visreps.dataloaders.neural import (
    _NSD_REGION_MAP, _NSD_SUBJECTS, _TVSD_SUBJECTS,
)

# preprocess_tvsd.py's subject -> monkey mapping.
_TVSD_MONKEYS = {0: "F", 1: "N"}


def spearman_brown(r: float, n: float) -> float:
    """Reliability of a measurement ``n`` times as long as one with reliability r."""
    return float(n * r / (1 + (n - 1) * r))


def subject_ceiling(
    betas: np.ndarray, trial_stimulus: np.ndarray, *,
    method: str = "spearman", n_splits: int = 20, seed: int = 42,
) -> Dict:
    """Noise ceiling for one subject, from its own repeated presentations.

    Args:
        betas: (n_trials, n_voxels) responses.
        trial_stimulus: (n_trials,) stimulus index per trial. Stimuli presented
            only once are dropped, since they cannot be split.

    Returns:
        dict with ``ceiling``, the ``reliability`` behind it, and the counts
        used. Some papers plot the uncorrected reliability instead, so a figure
        should say which of the two it shows.
    """
    rng = np.random.default_rng(seed)
    stim, counts = np.unique(trial_stimulus, return_counts=True)
    stim, counts = stim[counts >= 2], counts[counts >= 2]
    if len(stim) < 3:
        raise ValueError("need >= 3 stimuli with repeated presentations")

    # (n_stimuli, max_reps) table of trial indices, -1 where a stimulus has fewer.
    max_reps = int(counts.max())
    trials = np.full((len(stim), max_reps), -1, dtype=np.int64)
    for row, s in enumerate(stim):
        hits = np.flatnonzero(trial_stimulus == s)
        trials[row, : len(hits)] = hits

    # Halves are as large as the presentations allow: 1-vs-1 with 3 reps
    # (NSD), 15-vs-15 with 30 (TVSD). Bigger halves mean the Spearman-Brown
    # step below extrapolates less far, so the estimate rests on data.
    half = counts // 2
    slot = np.arange(max_reps)[None, :]
    in_a, in_b = slot < half[:, None], (slot >= half[:, None]) & (slot < 2 * half[:, None])

    scores = []
    for _ in range(n_splits):
        keys = np.where(slot < counts[:, None], rng.random(trials.shape), np.inf)
        shuffled = np.take_along_axis(trials, np.argsort(keys, axis=1), axis=1)
        picked = betas[np.maximum(shuffled, 0)]  # (n_stimuli, max_reps, n_features)
        halves = [
            torch.as_tensor(
                (picked * mask[:, :, None]).sum(axis=1) / half[:, None],
                dtype=torch.float32)
            for mask in (in_a, in_b)
        ]
        scores.append(compute_rdm_correlation(
            compute_rdm(halves[0]), compute_rdm(halves[1]),
            correlation=method.capitalize(),
        ))

    # Equal halves, so this is the plain Spearman-Brown formula rather than an
    # unequal-halves approximation: from a `half`-trial average up to the
    # `counts`-trial average that models are scored against.
    r_half = float(np.mean(scores))
    reliability = spearman_brown(r_half, float(counts.mean() / half.mean()))
    return {
        "ceiling": float(np.sqrt(max(reliability, 0.0))),
        "reliability": reliability,
        "r_half": r_half,
        "reps_per_half": float(half.mean()),
        "n_stimuli": int(len(stim)),
        "mean_reps": float(counts.mean()),
    }


def nsd_ceiling(
    region: str, *, method: str = "spearman", subjects=None,
    n_splits: int = 20, seed: int = 42,
    repetitions_file: str = "nsd_repetitions.pkl",
) -> Dict:
    """Mean within-subject noise ceiling for one NSD ROI.

    Restricted to the stimuli every subject saw -- the set evals.py scores
    models on -- so the ceiling and results.db are directly comparable.
    """
    subjects = list(subjects) if subjects is not None else list(_NSD_SUBJECTS)
    region_key = _NSD_REGION_MAP.get(region, region)

    path = os.path.join(utils.get_env_var("NSD_DATA_DIR"), repetitions_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run "
            "python scripts/preprocess_data/preprocess_nsd_repetitions.py first"
        )
    reps = utils.load_pickle(path)["data"][region_key]

    shared = sorted(set.intersection(*[
        set(int(i) for i in reps[s]["stimulus_ids"]) for s in subjects
    ]))

    per_subject = {}
    for subj in subjects:
        entry = reps[subj]
        stim_per_trial = entry["stimulus_ids"][entry["trial_stimulus"]]
        keep = np.isin(stim_per_trial, shared)
        per_subject[subj] = subject_ceiling(
            entry["betas"][keep],
            np.searchsorted(shared, stim_per_trial[keep]),
            method=method, n_splits=n_splits, seed=seed,
        )

    ceilings = np.array([per_subject[s]["ceiling"] for s in subjects])
    return {
        "ceiling": float(ceilings.mean()),
        "sem": float(ceilings.std(ddof=1) / np.sqrt(len(ceilings))),
        "mean_reliability": float(np.mean([per_subject[s]["reliability"] for s in subjects])),
        "per_subject": {int(s): per_subject[s] for s in subjects},
        "n_stimuli": len(shared),
        "n_splits": n_splits,
        "method": method.lower(),
        "region": region,
        "neural_dataset": "nsd",
    }


def tvsd_ceiling(
    region: str, *, method: str = "spearman", subjects=None,
    n_splits: int = 20, seed: int = 42,
) -> Dict:
    """Mean within-subject noise ceiling for one TVSD region.

    TVSD's test set is 100 stimuli x 30 repetitions -- small enough to load
    straight from the bonner package, so there is no cached repetition file to
    keep in sync. Electrodes are the reliability-filtered set stored in
    fmri_responses.pkl, matched by neuroid id, so the ceiling covers exactly
    the electrodes evals.py scores.

    With 2 monkeys there is no across-subject alternative here: a leave-one-out
    ceiling needs at least 3 subjects.
    """
    from bonner.datasets.papale2025_tvsd import load_normalized_data

    subjects = list(subjects) if subjects is not None else list(_TVSD_SUBJECTS)
    filtered = utils.load_pickle(
        os.path.join("datasets", "neural", "tvsd", "fmri_responses.pkl"))

    per_subject = {}
    for subj in subjects:
        electrodes = filtered[region][subj]["test"].coords["neuroid"].values
        reps = (
            load_normalized_data(monkey=_TVSD_MONKEYS[subj], train=False)
            .unstack("presentation")
            .transpose("stimulus", "repetition", "neuroid")
            .values[:, :, electrodes]
        )  # (n_stimuli, n_repetitions, n_electrodes)
        n_stimuli, n_reps, n_elec = reps.shape
        per_subject[subj] = subject_ceiling(
            reps.reshape(n_stimuli * n_reps, n_elec),
            np.repeat(np.arange(n_stimuli), n_reps),
            method=method, n_splits=n_splits, seed=seed,
        )

    ceilings = np.array([per_subject[s]["ceiling"] for s in subjects])
    return {
        "ceiling": float(ceilings.mean()),
        "sem": float(ceilings.std(ddof=1) / np.sqrt(len(ceilings))) if len(ceilings) > 1 else float("nan"),
        "mean_reliability": float(np.mean([per_subject[s]["reliability"] for s in subjects])),
        "per_subject": {int(s): per_subject[s] for s in subjects},
        "n_stimuli": int(per_subject[subjects[0]]["n_stimuli"]),
        "n_splits": n_splits,
        "method": method.lower(),
        "region": region,
        "neural_dataset": "tvsd",
    }
