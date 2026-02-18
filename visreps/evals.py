import logging, torch
import pandas as pd
from omegaconf import OmegaConf
from visreps.utils import console, rprint, save_results
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


def _build_header(cfg):
    """Build a compact one-line summary header for eval output."""
    analysis = cfg.get("analysis", "rsa").upper()
    seed = cfg.get("seed", "?")
    seed_letter = get_seed_letter(seed) if isinstance(seed, int) else "?"
    cfg_id = cfg.get("cfg_id", "?")
    epoch = cfg.get("epoch", "?")
    neural_dataset = cfg.get("neural_dataset", "?").upper()
    region = cfg.get("region", "")
    subject_idx = cfg.get("subject_idx", "")

    parts = [f"{analysis} eval"]
    parts.append(f"cfg{cfg_id}{seed_letter} epoch {epoch}")
    if region and str(region).upper() != "N/A":
        parts.append(f"{neural_dataset} {region}")
    else:
        parts.append(neural_dataset)
    if subject_idx != "" and str(subject_idx).upper() != "N/A":
        parts.append(f"subj {subject_idx}")
    parts.append(f"seed {seed}")
    return " | ".join(parts)


# ───────────────────────── eval ──────────────────────────
def eval(cfg):
    """Return DataFrame of neural-alignment scores."""

    verbose = cfg.get("verbose", False)

    # ── CONFIG & DEVICE ─────────────────────────────────
    if cfg.load_model_from == "checkpoint":
        cfg = _load_cfg(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print compact header
    rprint(f"\n  {_build_header(cfg)}\n", style="info")

    # ── MODEL & FEATURE EXTRACTOR ───────────────────────
    model = mutils.load_model(cfg, dev, verbose=verbose)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

    # ── DATA & ACTIVATIONS ─────────────────────────────
    neural_data, dl = get_neural_loader(cfg)
    rprint(f"  ✓ {cfg.neural_dataset.upper()} data loaded", style="success")

    acts, ids = mutils.get_activations(model, dl, dev, cfg.apply_srp)
    if cfg.get("reconstruct_from_pcs"):
        acts = reconstruct_from_pcs(acts, cfg.pca_k)

    acts_aligned, neural_aligned = prepare_data_for_alignment(
        cfg, acts, neural_data, ids
    )

    # ── ALIGNMENT & SAVE ───────────────────────────────
    alignment_scores = compute_neural_alignment(
        cfg, acts_aligned, neural_aligned, verbose=verbose
    )
    results = pd.DataFrame(alignment_scores)

    if cfg.get("log_expdata"):
        save_results(results, cfg)

    return results
