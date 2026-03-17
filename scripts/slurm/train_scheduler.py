import json
import os
import subprocess
from itertools import product
from pathlib import Path

# =============================================================================
# USER CONFIGURATION - Modify these for your experiments
# =============================================================================

BASE_CONFIG = "configs/train/cluster_base.json"

PARAM_GRID = {
    "seed": [1],
    "model_class": ["standard_model"],
    "model_name": ["VGG16", "ResNet50", "ViTBase", "ConvNeXt_Base"],
    "pretrained_dataset": ["none"],
    "pca_labels": [True],
    "pca_n_classes": [8, 32],
    "pca_labels_folder": ["pca_labels_clip"],
    "log_checkpoints": [True],
}

# Used when pca_labels is False
DEFAULT_CHECKPOINT_DIR = "default"

SLURM_CONFIG = {
    "job-name": "visreps",
    "output": "scripts/slurm/slurm_logs/%j.out",
    "error": "scripts/slurm/slurm_logs/%j.err",
    "ntasks": "1",
    "cpus-per-task": "32",
    "gres": "gpu:1",
    "time": "8:00:00",
    "partition": "a100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

# =============================================================================
# INTERNAL
# =============================================================================


def get_checkpoint_dir(params):
    """Derive checkpoint_dir from pca_labels_folder when pca_labels=True.

    CustomCNN (custom_model) keeps the existing convention:
        "pca_labels_clip" -> "clip_pca"

    Standard architectures prepend the model name:
        ResNet50 + "pca_labels_clip" -> "resnet50_clip_pca"
    """
    if params.get("pca_labels"):
        folder = params.get("pca_labels_folder", "")
        base = folder.removeprefix("pca_labels_")
        model_class = params.get("model_class", "custom_model")
        if model_class == "standard_model":
            model_name = params.get("model_name", "").lower()
            return f"{model_name}_{base}_pca"
        return f"{base}_pca"
    return DEFAULT_CHECKPOINT_DIR


def build_overrides(params):
    """Convert params dict to CLI override strings."""
    overrides = [f"{k}={json.dumps(v)}" for k, v in params.items()]
    overrides.append(f"checkpoint_dir={json.dumps(get_checkpoint_dir(params))}")
    return overrides


def generate_slurm_script(overrides):
    """Generate SLURM batch script content."""
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items()]
    lines += [
        "",
        "source .venv/bin/activate",
        'echo "Running on: $(hostname)"',
        "nvidia-smi",
        "",
        f"python -m visreps.run --mode train --config {BASE_CONFIG} --override " + " ".join(overrides),
        "deactivate",
    ]
    return "\n".join(lines)


def iter_param_combinations():
    """Yield all parameter combinations as dicts."""
    keys = list(PARAM_GRID.keys())
    for values in product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def main():
    Path("scripts/slurm/slurm_logs").mkdir(parents=True, exist_ok=True)
    Path("scripts/slurm/tmp").mkdir(parents=True, exist_ok=True)

    combinations = list(iter_param_combinations())
    print(f"Submitting {len(combinations)} SLURM jobs\n")

    for i, params in enumerate(combinations, start=1):
        overrides = build_overrides(params)
        script_path = f"scripts/slurm/tmp/train_job_{i}.sh"

        with open(script_path, "w") as f:
            f.write(generate_slurm_script(overrides))

        print(f"Job {i}:")
        for o in overrides:
            print(f"  {o}")
        print()

        subprocess.run(["sbatch", script_path])
        os.remove(script_path)


if __name__ == "__main__":
    main()
