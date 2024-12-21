import numpy as np
import os
import pandas as pd
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from omegaconf import OmegaConf
from typing import Dict, List, Tuple

from visreps.dataloaders.neural import create_nsd_dataloader, load_nsd_data
from visreps.utils import extract_activations, save_results
import visreps.metrics as metrics
from visreps.models.model_loader import load_model
from visreps.dataloaders.obj_cls import get_transform


def setup_feature_extractor(model: torch.nn.Module, cfg: Dict) -> torch.nn.Module:
    """Setup model for feature extraction"""
    return_nodes = OmegaConf.to_container(cfg.get("return_nodes", {}), resolve=True)
    if not return_nodes:
        raise ValueError("return_nodes must be specified in config")
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes
    print(f"Extracting features from layers: {list(return_nodes.keys())}")
    return create_feature_extractor(model, return_nodes=return_nodes)


def prepare_neural_dataset_loader(cfg: Dict) -> Tuple[Dict, torch.utils.data.DataLoader]:
    """Prepare neural dataset and its dataloader based on config
    
    Args:
        cfg: Configuration dictionary with:
            - neural_dataset: Which dataset to load ('nsd', etc.)
            - Other dataset-specific parameters
    
    Returns:
        Tuple of (neural_responses, dataloader) where:
            - neural_responses: Dict mapping stimulus IDs to neural responses
            - dataloader: DataLoader for the corresponding stimuli
    """
    dataset = cfg.get('neural_dataset', 'nsd')  # Default to NSD for backward compatibility
    
    # Select appropriate data loading function
    if dataset == 'nsd':
        neural_data, stimuli = load_nsd_data(cfg)
        dataloader_fn = create_nsd_dataloader
    # Add more datasets here as needed
    # elif dataset == 'vim1':
    #     neural_data, stimuli = load_vim1_data(cfg)
    #     dataloader_fn = create_vim1_dataloader
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: ['nsd']")
    
    if not neural_data or not stimuli:
        raise ValueError(f"Failed to load {dataset} data")
    
    # Create dataloader using dataset-specific function
    dataloader = dataloader_fn(
        stimuli, 
        transform=get_transform(image_size=224),
        batch_size=cfg.batchsize, 
        num_workers=cfg.num_workers
    )
    print(f"Loaded {dataset} data with {len(stimuli)} stimuli")
    return neural_data, dataloader


def compute_neural_alignment(activations_dict: Dict[str, torch.Tensor], 
                           neural_data: Dict[str, np.ndarray],
                           keys: List[str],
                           cfg: Dict) -> pd.DataFrame:
    """Compute neural alignment scores for each layer using specified metric"""
    results = []
    neural_responses = np.array([neural_data[str(key)] for key in keys])
    metric_name = cfg.get('metric', 'rsa')
    
    # Get metric function based on config
    metric_fn = {
        'rsa': metrics.calculate_rsa_score,
    }.get(metric_name)
    
    if not metric_fn:
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(metric_fn.keys())}")
    
    print(f"Computing {metric_name.upper()} scores...")
    for layer, activations in activations_dict.items():
        # Flatten activations if needed
        activations = activations.flatten(start_dim=1) if activations.ndim > 2 else activations
        score = metric_fn(neural_responses, activations.cpu().numpy())
        print(f"Layer {layer:<20} {metric_name.upper()} Score: {score:.4f}")
        
        # Store results
        result = {
            "layer": layer,
            f"{metric_name}_score": score,
            "metric": metric_name
        }
        result.update(cfg if isinstance(cfg, dict) else vars(cfg))
        results.append(result)
    
    return pd.DataFrame(results)


def eval(cfg) -> pd.DataFrame:
    """Evaluate model's neural alignment with brain data"""
    print("\nStarting model evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. Setup
        print("Loading model...")
        model = load_model(cfg).to(device)
        model = setup_feature_extractor(model, cfg)
        
        # 2. Load data
        print("Loading data...")
        neural_data, dataloader = prepare_neural_dataset_loader(cfg)
        
        # 3. Compute alignment
        print("Computing neural alignment...")
        activations_dict, keys = extract_activations(model, dataloader, device)
        results_df = compute_neural_alignment(activations_dict, neural_data, keys, cfg)
        
        # 4. Save and return
        results_path = save_results(results_df, cfg, result_type='neural_alignment')
        print(f"\nResults saved to: {results_path}")
        
        return results_df
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise
