import json
import subprocess
from itertools import product

# Configuration file paths
BASE_CONFIG = "configs/train/base.json"

# Define the parameter grid for training
PARAM_GRID = {
    "exp_name": ["dropout"],
    "arch.nonlinearity": ["sigmoid"],
    "arch.dropout": [0.0, 0.3],
    "arch.batchnorm": [True],
    "learning_rate": [0.0005],
    "weight_decay": [0.001],
    "pca_labels": [False],
    "pca_n_classes": [2],
}

def flatten_config(config, parent_key="", sep="."):
    """Recursively flattens a nested configuration dictionary."""
    flat = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, new_key, sep))
        else:
            flat[new_key] = v
    return flat

def load_config(filepath):
    """Loads a JSON configuration from the given file."""
    with open(filepath, "r") as file:
        return json.load(file)

def main():
    # Generate all combinations of grid parameters
    param_names = list(PARAM_GRID.keys())
    param_combos = list(product(*PARAM_GRID.values()))

    total_runs = len(param_combos)
    print(f"Running {total_runs} training configurations")

    # Run each combination
    for combo in param_combos:
        # Create overrides from the parameter combination
        overrides = [f"{name}={json.dumps(value)}" for name, value in zip(param_names, combo)]
        
        # Add required mode override
        overrides.append("mode=train")
        
        cmd = ["python", "-m", "visreps.run",
               "--config", BASE_CONFIG,
               "--override"] + overrides

        cmd_str = " ".join(cmd)
        print(f"\nExecuting: {cmd_str}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main() 