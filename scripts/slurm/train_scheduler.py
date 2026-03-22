import json
import os
import subprocess
from itertools import product
from pathlib import Path

# =============================================================================
# USER CONFIGURATION
# =============================================================================

BASE_CONFIG = "configs/train/base.json"

# Human-friendly name -> architecture config path
MODELS = {
    "CustomCNN":     "configs/train/architectures/custom_cnn.json",
    "AlexNet":       "configs/train/architectures/alexnet.json",
    "ResNet18":      "configs/train/architectures/resnet18.json",
    "ResNet50":      "configs/train/architectures/resnet50.json",
    "VGG16":         "configs/train/architectures/vgg16.json",
    "ViTBase":       "configs/train/architectures/vit_b_16.json",
    "ConvNeXt_Base": "configs/train/architectures/convnext_base.json",
}

# Everything listed here gets submitted. Comment out what you don't want.
EXPERIMENTS = [
    {
        "name": "coarsegrain",
        "models": ["ResNet50", "ConvNeXt_Base", "ViTBase"],
        "seeds": [1],
        "pca_n_classes": [2, 4, 8, 16, 64],
        "pca_labels_folder": "pca_labels_clip",
    },
    {
        "name": "standard",
        "models": ["ResNet50", "ConvNeXt_Base", "ViTBase"],
        "seeds": [1],
    },
]

SLURM_CONFIG = {
    "job-name": "visreps",
    "output": "scripts/slurm/slurm_logs/%j.out",
    "error": "scripts/slurm/slurm_logs/%j.err",
    "ntasks": "1",
    "cpus-per-task": "32",
    "gres": "gpu:1",
    "time": "16:00:00",
    "partition": "a100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

# =============================================================================
# INTERNAL
# =============================================================================

DEFAULT_CHECKPOINT_DIR = "default"


def get_checkpoint_dir(model_name, pca_labels, pca_labels_folder=None):
    """Derive checkpoint_dir from model name and PCA config.

    CustomCNN keeps the existing convention:
        pca_labels=True  + "pca_labels_alexnet" -> "alexnet_pca"
        pca_labels=False                        -> "default"

    Standard architectures prepend the model name:
        ResNet50 + "pca_labels_clip" -> "resnet50_clip_pca"
        ResNet50 + pca_labels=False  -> "resnet50_default"
    """
    is_custom = model_name == "CustomCNN"
    if pca_labels:
        base = (pca_labels_folder or "").removeprefix("pca_labels_")
        if not is_custom:
            return f"{model_name.lower()}_{base}_pca"
        return f"{base}_pca"
    if not is_custom:
        return f"{model_name.lower()}_{DEFAULT_CHECKPOINT_DIR}"
    return DEFAULT_CHECKPOINT_DIR


def generate_slurm_script(base_config, arch_config, overrides):
    """Generate SLURM batch script content."""
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH --{k}={v}" for k, v in SLURM_CONFIG.items()]
    lines += [
        "",
        "source .venv/bin/activate",
        'echo "Running on: $(hostname)"',
        "nvidia-smi",
        "",
        f"python -m visreps.run --mode train --config {base_config} {arch_config} --override log_checkpoints=true " + " ".join(overrides),
        "deactivate",
    ]
    return "\n".join(lines)


def build_jobs(experiments):
    """Expand experiments into a flat list of (model_name, arch_config, overrides) tuples."""
    jobs = []
    for exp in experiments:
        name = exp["name"]
        pca_labels = "pca_n_classes" in exp
        pca_n_classes_list = exp.get("pca_n_classes", [None])
        if not isinstance(pca_n_classes_list, list):
            pca_n_classes_list = [pca_n_classes_list]
        pca_labels_folder = exp.get("pca_labels_folder", "pca_labels_alexnet")

        for model_name in exp["models"]:
            arch_config = MODELS[model_name]
            for seed, pca_n_classes in product(exp["seeds"], pca_n_classes_list):
                checkpoint_dir = get_checkpoint_dir(model_name, pca_labels, pca_labels_folder)

                overrides = [
                    f"seed={seed}",
                    f"pca_labels={json.dumps(pca_labels)}",
                    f"checkpoint_dir={json.dumps(checkpoint_dir)}",
                ]
                if pca_labels:
                    overrides.append(f"pca_n_classes={pca_n_classes}")
                    overrides.append(f"pca_labels_folder={pca_labels_folder}")

                jobs.append((name, model_name, arch_config, overrides))
    return jobs


def main():
    Path("scripts/slurm/slurm_logs").mkdir(parents=True, exist_ok=True)
    Path("scripts/slurm/tmp").mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(EXPERIMENTS)
    print(f"Submitting {len(jobs)} SLURM jobs\n")

    for job_num, (exp_name, model_name, arch_config, overrides) in enumerate(jobs, 1):
        script_path = f"scripts/slurm/tmp/train_job_{job_num}.sh"

        with open(script_path, "w") as f:
            f.write(generate_slurm_script(BASE_CONFIG, arch_config, overrides))

        print(f"Job {job_num} [{exp_name} | {model_name}]:")
        for o in overrides:
            print(f"  {o}")
        print()

        subprocess.run(["sbatch", script_path])
        os.remove(script_path)


if __name__ == "__main__":
    main()
