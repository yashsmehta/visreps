import json
import os
import subprocess
from itertools import product
from pathlib import Path

# =============================================================================
# USER CONFIGURATION - Modify these for your experiments
# =============================================================================

# Each model maps to its architecture-specific config (with correct optimizer,
# LR, scheduler, etc.) and an estimated wall-clock time for full ImageNet
# training at bs=32 on a single GPU.
MODEL_CONFIGS = {
    "ResNet50":      "configs/train/resnet50.json",
    "ViTBase":       "configs/train/vit_b_16.json",
    "ConvNeXt_Base": "configs/train/convnext_base.json",
}

PARAM_GRID = {
    "seed": [1],
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
    "time": "47:59:59",
    "partition": "a100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

# =============================================================================
# INTERNAL
# =============================================================================


def get_checkpoint_dir(model_name, params):
    """Derive checkpoint_dir from model name and pca_labels_folder.

    Standard architectures prepend the model name:
        ResNet50 + "pca_labels_clip" -> "resnet50_clip_pca"
        ResNet50 + pca_labels=False  -> "resnet50_default"

    CustomCNN (custom_model) keeps the existing convention:
        "pca_labels_clip" -> "clip_pca"
        pca_labels=False  -> "default"
    """
    if params.get("pca_labels"):
        folder = params.get("pca_labels_folder", "")
        base = folder.removeprefix("pca_labels_")
        if model_name:
            return f"{model_name.lower()}_{base}_pca"
        return f"{base}_pca"
    if model_name:
        return f"{model_name.lower()}_{DEFAULT_CHECKPOINT_DIR}"
    return DEFAULT_CHECKPOINT_DIR


def build_overrides(model_name, params):
    """Convert params dict to CLI override strings."""
    overrides = [f"{k}={json.dumps(v)}" for k, v in params.items()]
    overrides.append(f"checkpoint_dir={json.dumps(get_checkpoint_dir(model_name, params))}")
    return overrides


def generate_slurm_script(config_path, overrides):
    """Generate SLURM batch script content."""
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items()]
    lines += [
        "",
        "source .venv/bin/activate",
        'echo "Running on: $(hostname)"',
        "nvidia-smi",
        "",
        f"python -m visreps.run --mode train --config {config_path} --override " + " ".join(overrides),
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

    param_combos = list(iter_param_combinations())
    total_jobs = len(MODEL_CONFIGS) * len(param_combos)
    print(f"Submitting {total_jobs} SLURM jobs "
          f"({len(MODEL_CONFIGS)} models × {len(param_combos)} param combos)\n")

    job_num = 0
    for model_name, config_path in MODEL_CONFIGS.items():
        for params in param_combos:
            job_num += 1
            overrides = build_overrides(model_name, params)
            script_path = f"scripts/slurm/tmp/train_job_{job_num}.sh"

            with open(script_path, "w") as f:
                f.write(generate_slurm_script(config_path, overrides))

            print(f"Job {job_num} [{model_name}]:")
            print(f"  config: {config_path}")
            for o in overrides:
                print(f"  {o}")
            print()

            subprocess.run(["sbatch", script_path])
            os.remove(script_path)


if __name__ == "__main__":
    main()
