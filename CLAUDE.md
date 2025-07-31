# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install uv package manager and create virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Running Experiments
```bash
# Training models
python -m visreps.run --mode train --config configs/train/base.json

# Evaluating models
python -m visreps.run --mode eval --config configs/eval/base.json

# Override config parameters
python -m visreps.run --mode train --config configs/train/base.json --override learning_rate=0.01 num_epochs=20
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_training_pipeline.py

# Run tests with coverage
pytest tests/ --cov=visreps
```

### Development Tools
```bash
# Format code with black
black visreps/ scripts/ tests/

# Run linter
flake8 visreps/ scripts/ tests/

# Type checking
mypy visreps/
```

## Architecture Overview

### Core Components

1. **Entry Point** (`visreps/run.py`)
   - Main script that handles both training and evaluation modes
   - Loads configuration from JSON files in `configs/`
   - Routes to either `Trainer` or `evals.eval()`

2. **Training Pipeline** (`visreps/trainer.py`)
   - Manages model training for object classification
   - Handles PCA-based label sets (2-1000 classes) for granularity experiments
   - Supports checkpoint saving and wandb logging
   - Key methods: `train()`, `evaluate()`, `save_checkpoint()`

3. **Evaluation System** (`visreps/evals.py`)
   - Computes neural alignment using RSA (Representational Similarity Analysis)
   - Works with fMRI data (NSD dataset) and behavioral data (THINGS)
   - Supports sparse random projection for dimensionality reduction
   - Reconstructs representations from principal components

4. **Model Architecture** (`visreps/models/`)
   - `standard_cnn.py`: Pre-trained models (AlexNet, ResNet, CLIP)
   - `custom_cnn.py`: Custom CNN architectures with configurable layers
   - `utils.py`: Model loading, feature extraction, activation collection
   - Feature extraction from intermediate layers (conv1-5, fc1-2)

5. **Analysis Tools** (`visreps/analysis/`)
   - `rsa.py`: Representational similarity analysis
   - `alignment.py`: Neural alignment computation
   - `compute_eigenspectra.py`: Eigenvalue analysis
   - `regression/`: Linear regression tools for brain prediction
   - `metrics/`: CKA, correlation coefficient, R² score implementations

6. **Data Loading** (`visreps/dataloaders/`)
   - `obj_cls.py`: ImageNet and Tiny-ImageNet loaders with PCA labels support
   - `neural.py`: NSD fMRI data and THINGS behavioral data loaders

### Configuration System

- JSON-based configs in `configs/train/` and `configs/eval/`
- Key parameters:
  - `pca_labels`: Enable PCA-based label grouping
  - `pca_n_classes`: Number of coarse categories (2, 4, 8, 16, 32, 64)
  - `model_class`: "standard_cnn" or "custom_cnn"
  - `return_nodes`: Layers to extract features from
  - `neural_dataset`: "nsd" for fMRI or "things" for behavioral data

### Model Checkpoints

- Saved in `model_checkpoints/` with structure:
  - `{dataset}_{variant}/cfg{id}{seed_letter}/`
  - Contains `config.json` and `checkpoint_epoch_{n}.pth`
- Naming convention: `imagenet_pca` for coarse models, `imagenet_1k` for standard

### Key Workflows

1. **Training with PCA labels**:
   - Generate coarse labels using scripts in `scripts/`
   - Configure `pca_labels=true` and `pca_n_classes` in config
   - Train model with reduced label granularity

2. **Neural Alignment Analysis**:
   - Load trained checkpoint or pre-trained model
   - Extract representations for stimulus images
   - Compute RSMs and correlate with brain/behavioral data
   - Analyze per-region and per-subject alignment

3. **Representation Analysis**:
   - Extract activations from multiple layers
   - Apply dimensionality reduction (PCA, sparse random projection)
   - Compare models trained with different granularities
   - Evaluate on in-distribution and out-of-distribution stimuli