import logging, torch
import pandas as pd
from omegaconf import OmegaConf
from visreps.utils import rprint, save_results
from visreps.utils import get_seed_letter
import visreps.models.utils as mutils
from visreps.dataloaders.neural import get_neural_loader
from visreps.analysis.alignment import (
    compute_neural_alignment,
    prepare_data_for_alignment,
)
from visreps.analysis.reconstruct_from_pcs import reconstruct_from_pcs

logger = logging.getLogger(__name__)


# ──────────────────────── helpers ────────────────────────
def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (drops `mode`)."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(path)
    epoch = int(cfg.checkpoint_model.split('_')[-1].split('.')[0])
    base.epoch = epoch
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


# ───────────────────────── eval ──────────────────────────
def eval(cfg):
    """Return DataFrame of neural-alignment scores."""

    # 1 ── CONFIG & DEVICE ─────────────────────────────────
    rprint("\n[1/4] Config + device", style="info")
    if cfg.load_model_from == "checkpoint":
        cfg = _load_cfg(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 ── MODEL & FEATURE EXTRACTOR ───────────────────────
    rprint("\n[2/4] Load model", style="info")
    model = mutils.configure_feature_extractor(cfg, mutils.load_model(cfg, dev))

    # 3 ── DATA & ACTIVATIONS ─────────────────────────────
    rprint("\n[3/4] Data + activations", style="info")
    neural_data, dl = get_neural_loader(cfg)
    acts, ids = mutils.get_activations(model, dl, dev, cfg.apply_srp)
    if cfg.get("reconstruct_from_pcs"):
        acts = reconstruct_from_pcs(acts, cfg.pca_k)

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
