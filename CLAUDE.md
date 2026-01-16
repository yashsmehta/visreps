# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always activate the virtual environment before running any commands:
```bash
source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate
```

## Project Overview

This repository investigates whether fine-grained category supervision is necessary for brain-model alignment in visual representation learning. It trains CNNs (primarily AlexNet variants) on ImageNet with varying label granularity (2 to 1000 classes) using PCA-based coarse-grained labels, then evaluates alignment with human visual representations via fMRI (NSD dataset) and behavioral data (THINGS).

## Common Commands

### Training
```bash
# Basic training with default config
python -m visreps.run --mode train

# Training with specific config
python -m visreps.run --mode train --config configs/train/base.json

# Training with parameter overrides
python -m visreps.run --mode train --override pca_labels=true pca_n_classes=32 seed=1

# Submit SLURM jobs for grid search
python slurm/train_scheduler.py
```

### Evaluation
```bash
# Evaluate model against neural data
python -m visreps.run --mode eval

# Evaluate with specific config and overrides
python -m visreps.run --mode eval --config configs/eval/base.json --override neural_dataset=nsd analysis=rsa
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_training_pipeline.py

# Run with verbose output
pytest -v

# Run single test
pytest tests/test_run.py::test_run_main_dispatch -v
```

### Linting and Formatting
```bash
black visreps/
flake8 visreps/
mypy visreps/
```

## Architecture

### Entry Point and Configuration Flow
- `visreps/run.py`: Single entry point that routes to training (`Trainer`) or evaluation (`evals.eval()`) based on `--mode`
- Config files in `configs/` use JSON with OmegaConf for hierarchical merging
- CLI overrides take precedence: `--override key=value key2=value2`
- Nested configs (`custom_model`/`standard_model` for train, `checkpoint`/`torchvision` for eval) are merged based on `model_class` or `load_model_from`

### Training Pipeline
- `visreps/trainer.py`: Training loop with checkpointing, mixed precision (AMP), and WandB integration
- `visreps/dataloaders/obj_cls.py`: ImageNet/TinyImageNet loaders with PCA label support
- `visreps/models/custom_model.py` and `standard_model.py`: CNN architectures with configurable layer freezing (`conv_trainable`, `fc_trainable` as binary strings like "11111")

### Evaluation Pipeline
- `visreps/evals.py`: Loads trained models, extracts activations, computes neural alignment
- `visreps/dataloaders/neural.py`: NSD fMRI and THINGS behavioral data loaders
- `visreps/analysis/rsa.py`: Representational Similarity Analysis
- `visreps/analysis/alignment.py`: Linear probes and encoding models
- Results saved to `logs/` as CSV files with file locking for concurrent writes

### Model Checkpoints
- Saved to `model_checkpoints/{checkpoint_dir}/cfg{id}{seed_letter}/`
- Seed letters: 1→a, 2→b, 3→c (via `get_seed_letter()`)
- Each checkpoint includes model weights, optimizer state, and `config.json`

### External Data Paths
Configured via `.env`:
- `IMAGENET_DATA_DIR`: ImageNet images
- `NSD_DATA_DIR`: Natural Scenes Dataset (fMRI)
- PCA labels stored locally in `datasets/obj_cls/imagenet/pca_labels_*/`

### Grid Search
- Grid configs in `configs/grids/` define parameter sweep arrays
- `slurm/train_scheduler.py` and `eval_scheduler.py` generate and submit SLURM jobs from grid configs

## Key Patterns

### Config Verification
The `ConfigVerifier` class in `visreps/utils.py` validates configs before execution. Valid values:
- `mode`: "train" or "eval"
- `dataset`: "imagenet", "tiny-imagenet", "imagenet-mini-10/50/200"
- `neural_dataset`: "nsd", "things", "nsd_synthetic", "cusack"
- `analysis`: "rsa", "encoding_score"
- `seed`: 1, 2, or 3 (for eval mode)

### Layer Extraction
Use `return_nodes` config to specify which layers to extract activations from (e.g., `["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]`).

### Sparse Random Projection
Set `apply_srp: true` to apply sparse random projection for dimensionality reduction before alignment computation.
