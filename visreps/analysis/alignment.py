import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import logging
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

from visreps.analysis.rsa import compute_rsa_alignment

logger = logging.getLogger(__name__)


def compute_neural_alignment(
    cfg: DictConfig,
    acts_aligned: Dict[str, torch.Tensor],
    neural_aligned: torch.Tensor,
) -> List[dict]:
    """
    Compute neural alignment scores using RSA.

    Args:
        cfg (DictConfig): Configuration object.
        acts_aligned (Dict[str, torch.Tensor]): Model activations aligned to neural data.
        neural_aligned (torch.Tensor): Aligned neural or behavioral data.

    Returns:
        List[dict]: List of alignment scores per layer, each as a dict.
    """
    return compute_rsa_alignment(cfg, acts_aligned, neural_aligned)


def prepare_data_for_alignment(
    cfg: DictConfig,
    acts_raw: Dict[str, torch.Tensor],
    neural_data_raw: Dict[str, np.ndarray],
    keys: List[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Align and aggregate model activations and neural/behavioral data for alignment analysis.

    Supports 'nsd' (aligns by stimulus key) and 'things' (aggregates by concept).

    Args:
        cfg (DictConfig): Configuration object with dataset info.
        acts_raw (Dict[str, torch.Tensor]): Raw model activations keyed by layer.
        neural_data_raw (Dict[str, np.ndarray]): Raw neural/behavioral data keyed by stimulus or concept.
        keys (List[str]): List of stimulus identifiers.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: 
            - Aligned/aggregated activations (per layer).
            - Aligned/aggregated neural/behavioral data.
    """

    def _filter_nsd() -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        valid_idx = [i for i, k in enumerate(keys) if str(k) in neural_data_raw]
        if not valid_idx:
            logger.warning("No overlapping NSD keys.")
            return {}, torch.empty(0)
        neural = torch.as_tensor(np.stack([neural_data_raw[str(keys[i])] for i in valid_idx]))
        acts = {layer: act[valid_idx] for layer, act in acts_raw.items()}
        return acts, neural

    def _prepare_things() -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        concepts, idx_map = [], defaultdict(list)
        for i, k in enumerate(keys):
            c = "_".join(k.split("_")[:-1]) if "_" in k else k
            if c and c in neural_data_raw:
                if c not in concepts:
                    concepts.append(c)
                idx_map[c].append(i)
        if not concepts:
            logger.warning("No overlapping THINGS concepts found using derived names.")
            return {}, torch.empty(0)
        acts = {
            layer: torch.stack([act[idx_map[c]].float().mean(0).to(act.dtype) for c in concepts])
            for layer, act in acts_raw.items()
        }
        neural = torch.as_tensor(np.stack([neural_data_raw[c] for c in concepts], dtype=np.float32))
        return acts, neural

    dataset = cfg.neural_dataset.lower()
    if dataset == "nsd":
        return _filter_nsd()
    if dataset == "things":
        return _prepare_things()
    raise ValueError(f"Unsupported neural_dataset '{dataset}'")
