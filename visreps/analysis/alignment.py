import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import logging
from omegaconf import DictConfig

from visreps.analysis.rsa import compute_rsa
from visreps.analysis.encoding_score import compute_encoding_score

logger = logging.getLogger(__name__)


@dataclass
class AlignmentData:
    """Bundled activations and neural data for one split (train or test)."""
    activations: Dict[str, torch.Tensor]  # {layer_name: (n_stimuli, features)}
    neural: torch.Tensor                  # (n_stimuli, n_voxels)
    stimulus_ids: Optional[List[str]] = None  # ordered IDs matching rows
    concept_image_ids: Optional[Dict[str, List[str]]] = None  # THINGS: concept → [img_ids]


def _align_stimulus_level(acts_raw, targets, keys):
    """Align activations with neural targets by stimulus ID (shared helper).

    Returns (acts, neural, matched_ids) where matched_ids are the stimulus IDs
    in the same row order as acts/neural.
    """
    idx = [i for i, k in enumerate(keys) if str(k) in targets]
    matched_ids = [str(keys[i]) for i in idx]
    if not matched_ids:
        neural = torch.empty(0, dtype=torch.float32)
        acts = {l: a[:0] for l, a in acts_raw.items()}
        return acts, neural, matched_ids
    neural = torch.as_tensor(
        np.stack([targets[sid] for sid in matched_ids]), dtype=torch.float32
    )
    acts = {l: a[idx] for l, a in acts_raw.items()}
    return acts, neural, matched_ids


def prepare_traintest_alignment(
    cfg: DictConfig,
    acts_raw: Dict[str, torch.Tensor],
    neural_data_raw: Dict[str, Any],
    keys: List[str],
) -> Tuple[AlignmentData, AlignmentData]:
    """Align activations with train/test neural data, returning two AlignmentData objects.

    Used for stimulus-level datasets (NSD, TVSD) in the encoding_score path.

    Args:
        cfg: Config dict.
        acts_raw: {layer: (n_total_stimuli, features)} from feature extraction.
        neural_data_raw: Must contain "train" and "test" keys mapping stimulus IDs
            to neural response vectors.
        keys: Stimulus IDs corresponding to rows of acts_raw.

    Returns:
        (train, test): AlignmentData for each split.
    """
    train_acts, train_neural, train_ids = _align_stimulus_level(acts_raw, neural_data_raw["train"], keys)
    test_acts, test_neural, test_ids = _align_stimulus_level(acts_raw, neural_data_raw["test"], keys)
    train = AlignmentData(train_acts, train_neural, stimulus_ids=train_ids)
    test = AlignmentData(test_acts, test_neural, stimulus_ids=test_ids)

    logger.info(
        f"Prepared train/test alignment: {train.neural.size(0)} train, "
        f"{test.neural.size(0)} test samples."
    )
    return train, test


def compute_traintest_alignment(
    cfg: DictConfig,
    train: AlignmentData,
    test: AlignmentData,
    verbose: bool = False,
    re_extract_fn=None,
) -> List[dict]:
    """Dispatch to RSA or encoding score based on cfg.analysis.

    Args:
        re_extract_fn: Optional callback ``(layer_name, stimulus_ids=None) -> (tensor, ids)``
            for re-extracting a single layer without SRP. Passed to RSA only.
    """
    analysis = cfg.get("analysis", "rsa").lower()
    bootstrap = cfg.get("bootstrap", True)
    n_bootstrap = cfg.get("n_bootstrap", 1000)

    if analysis == "encoding_score" and cfg.get("neural_dataset", "").lower() == "things-behavior":
        raise ValueError(
            "Encoding score is not supported for things-behavior (behavioral embeddings "
            "have no voxels to predict). Use analysis=rsa instead."
        )

    if analysis == "rsa":
        n_select = cfg.get("n_select", None)  # None = use all train stimuli
        return compute_rsa(
            cfg, train, test,
            n_select=n_select, bootstrap=bootstrap,
            n_bootstrap=n_bootstrap, verbose=verbose,
            re_extract_fn=re_extract_fn,
        )
    elif analysis == "encoding_score":
        return compute_encoding_score(
            train, test,
            bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown analysis method: {analysis}")


def prepare_concept_alignment(
    cfg: DictConfig,
    acts_raw: Dict[str, torch.Tensor],
    neural_data_raw: Dict[str, Any],
    keys: List[str],
) -> AlignmentData:
    """Merge train/test THINGS images and average activations per concept.

    Combines train_image_ids and test_image_ids into a single set per concept,
    averages model activations across ALL images of each concept, and pairs
    with the concept-level behavioral embedding.

    Returns:
        AlignmentData with all ~1854 concepts (one row per concept).
        Also stores ``concept_image_ids`` mapping and ``stimulus_ids`` (concept
        order) for exact re-extraction.
    """
    key_to_idx = {k: i for i, k in enumerate(keys)}
    embeddings = neural_data_raw["train"]  # same embeddings in train and test

    # Merge image IDs from both splits
    train_ids = neural_data_raw["train_image_ids"]
    test_ids = neural_data_raw["test_image_ids"]
    all_concepts = list(train_ids.keys())

    concepts = []
    concept_acts = {l: [] for l in acts_raw}
    concept_image_ids = {}

    for concept in all_concepts:
        merged_img_ids = list(train_ids.get(concept, [])) + list(test_ids.get(concept, []))
        indices = [key_to_idx[sid] for sid in merged_img_ids if sid in key_to_idx]
        if not indices:
            continue

        concepts.append(concept)
        # Store which image IDs were actually found for this concept
        concept_image_ids[concept] = [sid for sid in merged_img_ids if sid in key_to_idx]
        for l, a in acts_raw.items():
            avg = a[indices].float().mean(0)
            concept_acts[l].append(avg)

    acts = {l: torch.stack(vs).to(acts_raw[l].dtype) for l, vs in concept_acts.items()}
    neural = torch.as_tensor(
        np.stack([embeddings[c] for c in concepts], dtype=np.float32)
    )

    logger.info(f"Prepared concept alignment: {len(concepts)} concepts (merged train+test images).")
    return AlignmentData(
        acts, neural,
        stimulus_ids=concepts,
        concept_image_ids=concept_image_ids,
    )


