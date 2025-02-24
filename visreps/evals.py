import torch
import visreps.models.utils as model_utils
from visreps.dataloaders.neural import get_neural_loader
from visreps.utils import save_results, rprint
from visreps.analysis.alignment import compute_neural_alignment
from omegaconf import OmegaConf

def eval(cfg):
    """Evaluate model's neural alignment with brain data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"\nStarting model evaluation on {device}", style="info")

    rprint("Loading model...", style="setup")
    if cfg.load_model_from == "checkpoint":
        train_cfg = OmegaConf.load(f"model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}/config.json")
        train_cfg.pop("mode")
        cfg.update(train_cfg)

    model = model_utils.load_model(cfg, device)
    model = model_utils.configure_feature_extractor(cfg, model)

    # Get neural data and compute alignment
    rprint("Loading neural data...", style="setup")
    neural_data, dataloader = get_neural_loader(cfg)

    rprint("Computing neural alignment...", style="info")
    activations_dict, keys = model_utils.get_activations(model, dataloader, device)
    results_df = compute_neural_alignment(cfg, activations_dict, neural_data, keys)

    if cfg.log_expdata and cfg.analysis != "cross_decomposition":
        save_results(results_df, cfg)

    rprint("Evaluation complete!", style="success")
    return results_df
