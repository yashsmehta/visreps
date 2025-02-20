import torch
import os
from omegaconf import OmegaConf
from visreps.models.utils import load_model

def test_model_weights_different():
    # Setup configs
    base_cfg = {
        'load_model_from': 'checkpoint',
        'exp_name': 'seed_test',
        'checkpoint_model': 'checkpoint_epoch_1.pth'
    }
    
    # Create configs for both models
    cfg1 = OmegaConf.create(base_cfg)
    cfg2 = OmegaConf.create(base_cfg)
    cfg1.cfg_id = 3 
    cfg2.cfg_id = 2 
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = load_model(cfg1, device, num_classes=10)
    model2 = load_model(cfg2, device, num_classes=10)
    
    # Compare weights
    all_equal = True
    param_diffs = []
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        if not torch.allclose(p1, p2):
            all_equal = False
            diff = (p1 - p2).abs().mean().item()
            param_diffs.append((name1, diff))
    
    if all_equal:
        print("WARNING: All model parameters are identical!")
    else:
        print("Models have different weights:")
        for name, diff in param_diffs:
            print(f"{name}: Mean absolute difference = {diff:.6f}")

if __name__ == "__main__":
    test_model_weights_different()
