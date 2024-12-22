import torch
from omegaconf import OmegaConf
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict

def setup_model(model: torch.nn.Module, cfg: Dict) -> torch.nn.Module:
    """Setup model for feature extraction
    
    Args:
        model: PyTorch model to extract features from
        cfg: Configuration dictionary containing:
            - return_nodes: List or dict of layer names to extract features from
    
    Returns:
        Feature extractor model that returns activations from specified layers
        
    Raises:
        ValueError: If return_nodes not specified in config
    """
    return_nodes = OmegaConf.to_container(cfg.get("return_nodes", {}), resolve=True)
    if not return_nodes:
        raise ValueError("return_nodes must be specified in config")
    
    # Convert list to dict if needed
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes
    print(f"Extracting features from layers: {list(return_nodes.keys())}")
    
    return create_feature_extractor(model, return_nodes=return_nodes) 


def get_activations(model, dataloader, device):
    """
    Extract activations from a model for a given dataloader.
    """
    model.eval()
    activations_dict = {}
    all_keys = []
    with torch.no_grad():
        for images, keys in dataloader:
            inputs = images.to(device)
            outputs = model(inputs)
            all_keys.extend(keys)
            for node_name, output in outputs.items():
                output = output.cpu()
                activations_dict.setdefault(node_name, []).append(output)
    for node_name in activations_dict:
        activations_dict[node_name] = torch.cat(activations_dict[node_name], dim=0)
    return activations_dict, all_keys
