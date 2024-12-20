import numpy as np
import os
import pandas as pd
import re
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from omegaconf import OmegaConf

from visreps.dataloaders.neural import create_nsd_dataloader
from visreps.utils import extract_activations
import visreps.metrics as metrics
from visreps.models.model_loader import load_model
import visreps.dataloaders.neural as neural
from visreps.dataloaders.obj_cls import get_transform


def evaluate_model(model, dataloader, neural_data, cfg, device):
    return_nodes = cfg.get("return_nodes")
    if not return_nodes:
        raise ValueError("return_nodes must be specified in the configuration")

    return_nodes = OmegaConf.to_container(return_nodes, resolve=True)
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes

    model = create_feature_extractor(model, return_nodes=return_nodes)
    activations_dict, keys = extract_activations(model, dataloader, device)
    neural_responses = np.array([neural_data[str(key)] for key in keys])

    results = []
    for layer, activations in activations_dict.items():
        activations = activations.flatten(start_dim=1) if activations.ndim > 2 else activations
        rsa_score = metrics.calculate_rsa_score(neural_responses, activations.cpu().numpy())

        result = {"layer": layer, "rsa_score": rsa_score}
        result.update(cfg if isinstance(cfg, dict) else vars(cfg))
        results.append(result)
        print(f"Layer {layer}: RSA Score = {rsa_score}")

    return results


def eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neural_data, stimuli = neural.load_nsd_data(cfg)
    transform = get_transform(image_size=224)
    dataloader = create_nsd_dataloader(stimuli, transform, batch_size=cfg.batchsize, num_workers=cfg.num_workers)

    model = load_model(cfg)
    model.to(device)
    results = evaluate_model(model, dataloader, neural_data, cfg, device)

    results_df = pd.DataFrame(results)
    
    # Create a more robust save directory structure
    exp_name = getattr(cfg, 'exp_name', 'default')
    model_class = getattr(cfg, 'model_class', 'unknown')
    save_dir = os.path.join('logs', exp_name, model_class)
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    return results_df
