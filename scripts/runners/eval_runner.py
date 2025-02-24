import os
import json
import subprocess
from itertools import product
from typing import Dict, Any

# Configuration
BASE_CONFIG_PATH = "configs/eval/base.json"
CHECKPOINTS_DIR = "model_checkpoints"

# Evaluation parameter grid
PARAM_GRID = {
    "exp_name": ["seeds"],
    "config_folders": ["cfg1", "cfg2"],
    "eval_epochs": [0, 10],
    "region": [
        "early visual stream",
        "ventral visual stream",
    ],
    "analysis": ["rsa"],
    "subject_idx": [0, 1],
    "return_nodes": [["conv4", "fc2"]],
}


def get_checkpoint_path(exp_name: str, config_folder: str, epoch: int) -> str:
    return os.path.join(
        CHECKPOINTS_DIR, exp_name, config_folder, f"checkpoint_epoch_{epoch}.pth"
    )


def run_evaluation(params: Dict[str, Any], checkpoint_path: str):
    """Run a single evaluation with the given parameters and checkpoint."""
    checkpoint_name = os.path.basename(checkpoint_path)
    config_id = params["config_folders"].replace("cfg", "")

    # Print configuration
    print(f"\nRunning evaluation for {checkpoint_name}:")
    print(f"  Experiment: {params['exp_name']}")
    print(f"  Config:     {params['config_folders']}")
    print(f"  Region:     {params['region']}")
    print(f"  Analysis:   {params['analysis']}")
    print(f"  Subject:    {params['subject_idx']}")

    # Create command with all necessary overrides
    overrides = [f"{k}={json.dumps(v)}" for k, v in params.items()] + [
        "mode=eval",
        "neural_dataset=nsd",
        "log_expdata=true",
        f"cfg_id={config_id}",
        "load_model_from=checkpoint",
        f"checkpoint_model={checkpoint_name}",
    ]

    cmd = [
        "python",
        "-m",
        "visreps.run",
        "--config",
        BASE_CONFIG_PATH,
        "--override",
    ] + overrides

    # Execute command
    subprocess.run(cmd)


def main():
    """Run all evaluations defined in the parameter grid."""
    # Generate parameter combinations
    keys = list(PARAM_GRID.keys())
    combinations = product(*(PARAM_GRID[key] for key in keys))

    for combo in combinations:
        params = dict(zip(keys, combo))

        # Get checkpoint path for the current epoch
        checkpoint_path = get_checkpoint_path(
            params["exp_name"], params["config_folders"], params["eval_epochs"]
        )

        run_evaluation(params, checkpoint_path)


if __name__ == "__main__":
    main()
