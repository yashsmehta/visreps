import torch
import visreps.models.utils as model_utils
from visreps.dataloaders.neural import get_neural_loader
from visreps.utils import save_results, rprint
from visreps.analysis.alignment import compute_neural_alignment

def eval(cfg):
    """Evaluate model's neural alignment with brain data using RSA"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"\nStarting model evaluation...device: {device}", style="info")
    
    # 1. Setup
    rprint("Loading model...", style="setup")
    model = model_utils.load_model(cfg, device)
    model = model_utils.configure_feature_extractor(cfg, model)
    
    # 2. Load data
    rprint("Loading neural data...", style="setup")
    neural_data, dataloader = get_neural_loader(cfg)
    
    # 3. Compute alignment and save results
    rprint("Computing alignment between neural data and model activations...", style="info")
    activations_dict, keys = model_utils.get_activations(model, dataloader, device)
    results_df = compute_neural_alignment(cfg, activations_dict, neural_data, keys)
    if cfg.log_expdata:
        save_results(results_df, cfg)
    rprint("Evaluation complete!", style="success")