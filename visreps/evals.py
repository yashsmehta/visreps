import logging, torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from visreps.utils import rprint, save_results
import visreps.models.utils as mutils
from visreps.dataloaders.neural import get_neural_loader
from visreps.analysis.alignment import (
    compute_neural_alignment,
    prepare_data_for_alignment,
)

logger = logging.getLogger(__name__)

# ──────────────────────── helpers ────────────────────────
def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (drops `mode`)."""
    path = f"model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}/config.json"
    base = OmegaConf.load(path)
    base.pop("mode", None)
    return OmegaConf.merge(base, cfg)

# ───────────────────────── eval ──────────────────────────
def eval(cfg):
    """Return DataFrame of neural-alignment scores."""

    # 1 ── CONFIG & DEVICE ─────────────────────────────────
    rprint("\n[1/4] Config + device", style="info")
    cfg  = _load_cfg(cfg)
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 ── MODEL & FEATURE EXTRACTOR ───────────────────────
    rprint("\n[2/4] Load model", style="info")
    model = mutils.configure_feature_extractor(cfg, mutils.load_model(cfg, dev))

    # 3 ── DATA & ACTIVATIONS ─────────────────────────────
    rprint("\n[3/4] Data + activations", style="info")
    neural_data, dl   = get_neural_loader(cfg)
    print("Loaded neural data and dataloader")
    acts, ids = mutils.get_activations(
        model,
        dl,
        dev,
        cfg.apply_srp
    )
    print("Extracted model activations and got stimuli ids")

    acts_aligned, neural_aligned = prepare_data_for_alignment(
        cfg, acts, neural_data, ids
    )

    # 4 ── ALIGNMENT & SAVE ───────────────────────────────
    rprint("\n[4/4] Alignment + save", style="info")
    alignment_scores = compute_neural_alignment(cfg, acts_aligned, neural_aligned)
    results = pd.DataFrame(alignment_scores)

    if cfg.get("log_expdata"):
        save_results(results, cfg)

    return results