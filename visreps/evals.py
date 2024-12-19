import os
import glob
import torch
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
import re

import visreps.dataloaders.neural as neural
import visreps.metrics as metrics
from visreps.dataloaders.obj_cls import get_transform
from visreps.dataloaders.neural import create_nsd_dataloader
from visreps.utils import extract_activations
from visreps.models.model_loader import load_model


def evaluate_model(model, dataloader, neural_data, cfg, device):
    return_nodes = getattr(cfg, "return_nodes", None)
    if return_nodes is None:
        raise ValueError("No return_nodes specified in the configuration")

    if isinstance(return_nodes, list):
        return_nodes = {node: node for node in return_nodes}
    elif not isinstance(return_nodes, dict):
        raise ValueError("return_nodes must be either a list or a dict")
    
    model = create_feature_extractor(model, return_nodes=return_nodes)
    activations_dict, keys = extract_activations(model, dataloader, device)
    neural_responses = np.array([neural_data[key] for key in keys])

    results = []
    for layer, activations in activations_dict.items():
        if len(activations.shape) > 2:
            activations = activations.view(activations.size(0), -1)
        rsa_score = metrics.calculate_rsa_score(neural_responses, activations.numpy())
        
        result = {
            "layer": layer,
            "rsa_score": rsa_score,
        }
        
        if isinstance(cfg, dict):
            result.update(cfg)
        else:
            result.update(vars(cfg))
        results.append(result)
        print(f"Layer {layer}: RSA Score = {rsa_score}")

    return results


def eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neural_data, stimuli = neural.load_benchmark_data(cfg)
    transform = get_transform(image_size=224)
    dataloader = create_nsd_dataloader(stimuli, transform, batch_size=32, num_workers=4)

    model = load_model(cfg)
    model.to(device)
    results = evaluate_model(model, dataloader, neural_data, cfg, device)

    results_df = pd.DataFrame(results)
    save_dir = os.path.dirname(cfg.model_name)
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    return results_df
