import numpy as np
import pandas as pd
import torch

import visreps.models.feature_extractor as feature_extractor
from visreps.models.model_loader import load_model
from visreps.dataloaders.neural import prepare_neural_dataset_loader
from visreps.utils import save_results
import visreps.metrics as metrics


def eval(cfg) -> pd.DataFrame:
    """Evaluate model's neural alignment with brain data"""
    print("\nStarting model evaluation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup
    print("Loading model...")
    model = load_model(cfg).to(device)
    model = feature_extractor.setup_model(model, cfg)
    
    # 2. Load data
    print("Loading neural data...")
    neural_data, dataloader = prepare_neural_dataset_loader(cfg)
    
    # 3. Compute alignment
    print("Computing neural data and model alignment...")
    activations_dict, keys = feature_extractor.get_activations(model, dataloader, device)
    results_df = metrics.compute_neural_alignment(activations_dict, neural_data, keys, cfg)
    
    # 4. Save and return
    results_path = save_results(results_df, cfg, result_type='neural_alignment')
    print(f"\nResults saved to: {results_path}")
    return results_df