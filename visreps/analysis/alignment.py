import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from visreps.analysis.rsa import compute_rsa_alignment
from visreps.analysis.regression.linear_regression import compute_linear_regression_alignment

def compute_neural_alignment(
    cfg: Dict,
    activations_dict: Dict[str, torch.Tensor], 
    neural_data: Dict[str, np.ndarray],
    keys: List[str],
) -> pd.DataFrame:
    """Compute neural alignment scores for each layer using specified analysis
    
    Args:
        cfg: Configuration dictionary with analysis settings
        activations_dict: Dict mapping layer names to activation tensors
        neural_data: Dict mapping stimulus IDs to neural responses
        keys: List of stimulus IDs in the order they were processed
        
    Returns:
        DataFrame containing alignment scores and metadata for each layer
    """
    # Convert neural data to tensor
    neural_responses = torch.tensor(np.array([neural_data[str(key)] for key in keys]))
    
    # Get analysis type from config
    analysis_type = cfg.get('analysis', 'rsa')
    
    # Map analysis types to their computation functions
    analysis_functions = {
        'rsa': compute_rsa_alignment,
        'linear_regression': compute_linear_regression_alignment,
        # Add more analysis types here as needed
        # 'cka': compute_cka_alignment,
    }
    
    if analysis_type not in analysis_functions:
        raise ValueError(f"Unknown analysis type: {analysis_type}. Available types: {list(analysis_functions.keys())}")
    
    # Compute alignment using specified analysis
    results = analysis_functions[analysis_type](cfg, activations_dict, neural_responses)
    
    # Create DataFrame and add config info
    results_df = pd.DataFrame(results)
    for result in results_df.to_dict('records'):
        result.update(cfg if isinstance(cfg, dict) else vars(cfg))
    
    return results_df