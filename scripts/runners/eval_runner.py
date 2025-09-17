import os
import json
import subprocess
from itertools import product
from typing import Dict, Any

# Configuration
BASE_CONFIG_PATH = "configs/eval/base.json"

# Evaluation parameter grid
PARAM_GRID = {
    # "checkpoint_dir": ["/data/ymehta3/clip_pca/"],
    # "checkpoint_dir": ["/data/rgautha1/for_yash/imagenet1k"],
    "checkpoint_dir": ["model_checkpoints/imagenet_mini_200_pca"],
    "cfg_id": [3, 4, 5, 6], # folder: cfg3 
    "seed": [1], # folder: cfg1a, cfg1b, cfg1c
    "results_csv": ["imagenet_mini_full-vs-pcs.csv"],
    "notes": [""],
    "compare_rsm_correlation": ["Spearman"],
    "reconstruct_from_pcs": [False],
    # "pca_k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "neural_dataset": ["nsd"],
    "subject_idx": [0],
    "region": ["early visual stream", "ventral visual stream"],
    "eval_checkpoint_at_epoch": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
}


def run_evaluation(params: Dict[str, Any]):
    """Run a single evaluation with the given parameters and checkpoint."""

    # Print configuration
    print(f"\nRunning evaluation for {params['checkpoint_model']}:")
    print(f"  Config:     {'cfg' + str(params['cfg_id'])}")
    print(f"  Results CSV: {params['results_csv']}")
    print(f"  Neural dataset: {params['neural_dataset']}")

    # Create command with all necessary overrides
    overrides = [f"{k}={json.dumps(v)}" for k, v in params.items()] + [
        "mode=eval",
        "log_expdata=true",
        "load_model_from=checkpoint",
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
        checkpoint_model = f"checkpoint_epoch_{params['eval_checkpoint_at_epoch']}.pth"
        params.pop("eval_checkpoint_at_epoch", None)
        params["checkpoint_model"] = checkpoint_model
        run_evaluation(params)


if __name__ == "__main__":
    main()
