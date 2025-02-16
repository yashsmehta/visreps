import os
import json
import glob
import subprocess
from itertools import product

# Configuration file paths
BASE_CONFIG = "configs/eval/base.json"
config_folder = "cfg1"
CHECKPOINT_DIR = f"model_checkpoints/tiny_imagenet_cnn/{config_folder}"

PARAM_GRID = {
    "region": ["early visual stream"],
    "analysis": ["rsa"],
    "subject_idx": [0],
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

def get_checkpoint_files(directory, pattern="checkpoint_epoch_*.pth"):
    """Returns a sorted list of checkpoint files matching the given pattern."""
    return sorted(glob.glob(os.path.join(directory, pattern)))

def main():
    training_cfg = load_config(os.path.join(CHECKPOINT_DIR, "config.json"))
    training_cfg.pop("checkpoint.checkpoint_path", None)
    training_cfg.pop("torchvision", None)
    training_cfg.pop("mode", None)

    flat_cfg = flatten_config(training_cfg)

    # Generate all combinations of grid parameters
    param_names = list(PARAM_GRID.keys())
    param_combos = list(product(*PARAM_GRID.values()))

    # Get all checkpoints
    checkpoint_files = get_checkpoint_files(CHECKPOINT_DIR)

    total_runs = len(param_combos) * len(checkpoint_files)
    print(f"Running {total_runs} configurations "
          f"({len(param_combos)} parameter combos × {len(checkpoint_files)} checkpoints)")

    # Run each combination for each checkpoint
    for checkpoint in checkpoint_files:
        checkpoint_rel = os.path.relpath(checkpoint)
        for combo in param_combos:
            # Start with flat config overrides and add grid params
            overrides = [f"{k}={json.dumps(v)}" for k, v in flat_cfg.items()]
            overrides += [f"{name}={value}" for name, value in zip(param_names, combo)]
            n_classes = 200 if not flat_cfg.get("pca_labels") else flat_cfg.get("pca_n_classes")
            overrides.append(f"n_classes={n_classes}")
            # Set the checkpoint path
            overrides.append(f"checkpoint_path={checkpoint_rel}")

            cmd = ["python", "-m", "visreps.run",
                   "--config", BASE_CONFIG,
                   "--override"] + overrides

            cmd_str = " ".join(cmd)
            print(f"\nExecuting: {cmd_str}")
            # Uncomment the line below to actually run the command
            subprocess.run(cmd)

if __name__ == "__main__":
    main()