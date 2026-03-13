"""Run pretrained ViT and CLIP evaluations across NSD, TVSD, and THINGS.

Populates results.db with pretrained model baselines for Figure 5.
Usage: python scripts/runners/pretrained_eval_runner.py [--models vit clip] [--datasets nsd tvsd things]
"""

import argparse
import subprocess
import json
import sys

MODELS = {
    "vit": {
        "model_name": "ViTBase",
        "pretrained_dataset": "imagenet1k",
    },
    "clip": {
        "model_name": "CLIP_ViT_L14",
        "pretrained_dataset": "openai",
    },
}

DATASETS = {
    "nsd": {
        "neural_dataset": "nsd",
        "subject_idx": [0, 1, 2, 3, 4, 5, 6, 7],
        "region": ["early visual stream", "ventral visual stream"],
    },
    # "tvsd": {
    #     "neural_dataset": "tvsd",
    #     "subject_idx": [0, 1],
    #     "region": ["V1", "V4", "IT"],
    # },
    # "things": {
    #     "neural_dataset": "things-behavior",
    # },
}

SHARED = {
    "load_model_from": "torchvision",
    "analysis": "rsa",
    "bootstrap": True,
    "log_expdata": True,
    "seed": 1,
    "batchsize": 64,
    "num_workers": 32,
}


def build_overrides(params):
    """Convert param dict to CLI override strings."""
    overrides = []
    for k, v in params.items():
        if isinstance(v, (bool, int, float)):
            overrides.append(f"{k}={json.dumps(v)}")
        elif isinstance(v, str):
            overrides.append(f"{k}={v}")
        else:
            overrides.append(f"{k}={json.dumps(v)}")
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    runs = []
    for model_key in args.models:
        for ds_key in args.datasets:
            params = {**SHARED, **MODELS[model_key], **DATASETS[ds_key]}
            runs.append((model_key, ds_key, params))

    print(f"Pretrained evaluation: {len(runs)} runs")
    print(f"  Models:   {args.models}")
    print(f"  Datasets: {args.datasets}\n")

    for idx, (model_key, ds_key, params) in enumerate(runs, 1):
        print(f"{'='*60}")
        print(f"Run {idx}/{len(runs)}: {model_key.upper()} on {ds_key.upper()}")
        print(f"{'='*60}")

        overrides = build_overrides(params)
        cmd = ["python", "-m", "visreps.run", "--mode", "eval", "--override"] + overrides

        if args.dry_run:
            print(f"  {' '.join(cmd)}\n")
        else:
            subprocess.run(cmd)
            print()


if __name__ == "__main__":
    main()
