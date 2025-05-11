import numpy as np
from typing import Dict, List, Tuple
import torch
import logging
from collections import defaultdict
from omegaconf import DictConfig
import pandas as pd

from visreps.analysis.rsa import compute_rsa_alignment

logger = logging.getLogger(__name__)

# ---------- helpers ----------
def _pca_reorder(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rows of `mat` by their score on the first principal component.
    Returns (reordered_mat, row_indices).
    Falls back to the original order if PCA fails.
    """
    try:
        centered = mat - mat.mean(0, keepdim=True)
        torch.manual_seed(42)
        _, _, V = torch.pca_lowrank(centered, q=50)
        if V.numel() == 0:
            raise RuntimeError("empty PC")
        pc1 = V[:, 0]
        projections = centered @ pc1
        order = torch.argsort(projections, stable=True)
        logger.info(f"Reordered neural data with {mat.size(0)} samples.")
        return mat[order], order

    except Exception as e:
        logger.warning(f"PCA reorder failed: {e}")
        return mat, torch.arange(mat.size(0))


# ---------- public API ----------

def compute_neural_alignment(
    cfg: DictConfig,
    acts_aligned: Dict[str, torch.Tensor],
    neural_aligned: torch.Tensor,
) -> List[dict]:
    """Thin wrapper around `compute_rsa_alignment` to keep the public signature unchanged."""
    return compute_rsa_alignment(cfg, acts_aligned, neural_aligned)


def prepare_data_for_alignment(
    cfg: DictConfig,
    acts_raw: Dict[str, torch.Tensor],
    neural_data_raw: Dict[str, np.ndarray],
    keys: List[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Align/aggregate activations and neural data for RSA analysis.
    Supports `nsd` (stimulus-level) and `things` (concept-level) pipelines.
    """
    dataset = cfg.neural_dataset.lower()

    # ---- NSD (stimulus-level alignment) ----
    if dataset == "nsd":
        idx = [i for i, k in enumerate(keys) if str(k) in neural_data_raw]
        neural = np.stack([neural_data_raw[str(keys[i])] for i in idx]).squeeze()
        neural = torch.as_tensor(neural)
        acts   = {l: a[idx] for l, a in acts_raw.items()}

    # ---- NSD Synthetic (stimulus-level alignment) ----
    elif dataset == "nsd_synthetic":
        # Read the CSV file to get the desired stimulus order
        csv_path = "datasets/neural/nsd_synthetic/extracted_coords.csv"
        order_df = pd.read_csv(csv_path) # Will raise FileNotFoundError if not found
        ordered_stimuli = order_df["stimulus"].tolist() # Will raise KeyError if 'stimulus' column is missing

        key_to_original_idx = {str(k): i for i, k in enumerate(keys)}
        idx = []
        for stim_id in ordered_stimuli:
            if str(stim_id) in neural_data_raw and str(stim_id) in key_to_original_idx:
                idx.append(key_to_original_idx[str(stim_id)])
        
        neural = np.stack([neural_data_raw[str(keys[i])] for i in idx]).squeeze()
        neural = torch.as_tensor(neural)
        acts   = {l: a[idx] for l, a in acts_raw.items()}

    # ---- THINGS (concept-level aggregation) ----
    elif dataset == "things":
        idx_map = defaultdict(list)
        for i, k in enumerate(keys):
            concept = "_".join(k.split("_")[:-1]) or k
            if concept in neural_data_raw:
                idx_map[concept].append(i)

        concepts = list(idx_map)
        acts = {
            l: torch.stack([acts_raw[l][idx_map[c]].float().mean(0) for c in concepts]).to(a.dtype)
            for l, a in acts_raw.items()
        }
        neural = torch.as_tensor(np.stack([neural_data_raw[c] for c in concepts], dtype=np.float32))

    else:
        raise ValueError(f"Unsupported neural_dataset '{dataset}'")

    # For nsd_synthetic, we use the CSV order, so PCA reorder is skipped.
    if dataset != "nsd_synthetic":
        neural, order = _pca_reorder(neural)
        acts   = {l: a[order] for l, a in acts.items()}
    
    logger.info(f"Prepared {dataset.upper()} data with {neural.size(0)} samples.")
    return acts, neural