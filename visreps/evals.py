import torch
import visreps.models.utils as model_utils
from visreps.dataloaders.neural import get_neural_loader
from visreps.utils import save_results
import visreps.metrics as metrics


def eval(cfg):
    """Evaluate model's neural alignment with brain data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting model evaluation...device: {device}")
    
    # 1. Setup
    print("Loading model...")
    model = model_utils.load_model(cfg, device)
    model = model_utils.configure_feature_extractor(model, cfg)
    
    # 2. Load data
    print("Loading neural data...")
    neural_data, dataloader = get_neural_loader(cfg)
    
    # 3. Compute alignment
    print("Computing neural data and model alignment...")
    activations_dict, keys = model_utils.get_activations(model, dataloader, device)
    results_df = metrics.compute_neural_alignment(activations_dict, neural_data, keys, cfg)
    
    # 4. Save and return
    results_path = save_results(results_df, cfg, result_type='neural_alignment')
    print(f"\nResults saved to: {results_path}")
    return