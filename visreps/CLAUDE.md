# visreps/ Package Documentation

This document provides an overview of the core `visreps/` package structure, which contains all the main code for training CNNs, extracting representations, and computing neural alignment scores.
You are NOT allowed to run any sudo commands on the system. And ask before deleting files or folders.

## Directory Structure

```
visreps/
├── run.py                  # Main entry point (routes to train/eval)
├── trainer.py              # Training loop and checkpointing
├── evals.py                # Evaluation orchestration
├── config.py               # ConfigDict class for config handling
├── utils.py                # Shared utilities (logging, optimization, file I/O)
│
├── analysis/               # Neural alignment and analysis methods
│   ├── alignment.py        # Main alignment computation (RSA/encoding)
│   ├── rsa.py              # Representational Similarity Analysis
│   ├── encoding_score.py   # Ridge regression encoding models (GPU)
│   ├── encoding_score_ray.py  # Ray-based parallel encoding (legacy)
│   ├── extract_representations.py  # Extract CNN activations
│   ├── sparse_random_projection.py  # SRP dimensionality reduction
│   ├── cross_decomposition.py  # CCA/PLS methods
│   ├── compute_eigenspectra.py  # Eigenspectrum analysis
│   ├── compute_twoNN_ID.py  # Intrinsic dimensionality
│   ├── reconstruct_from_pcs.py  # PC reconstruction
│   └── metrics/            # Similarity/distance metrics
│       ├── _cka.py         # Centered Kernel Alignment
│       ├── _corrcoef.py    # Correlation coefficient
│       └── _r2_score.py    # R-squared score
│
├── models/                 # CNN architectures and utilities
│   ├── custom_model.py     # Custom CNN implementations (AlexNet-like)
│   ├── standard_model.py   # Torchvision model wrappers
│   ├── ecnet.py            # ECNet architecture
│   ├── utils.py            # Model loading, checkpointing, feature extraction
│   └── nn_ops.py           # Neural network operations (pooling, activations)
│
└── dataloaders/            # Data loading for training and evaluation
    ├── obj_cls.py          # ImageNet/classification datasets (with PCA labels)
    └── neural.py           # Neural datasets (NSD fMRI, THINGS behavior)
```

## Core Files Overview

### Entry Points

#### `run.py`
**Purpose**: Single entry point for all training and evaluation runs.

**Key Functions**:
- `main()`: Parses CLI args (`--mode`, `--config`, `--override`), loads config, dispatches to `Trainer` or `evals.eval()`

**Usage**:
```bash
python -m visreps.run --mode train --config configs/train/base.json
python -m visreps.run --mode eval --override seed=1 analysis=rsa
```

---

### Training

#### `trainer.py`
**Purpose**: Handles the complete training loop with checkpointing, mixed precision, and WandB logging.

**Key Class**: `Trainer`
- `__init__(cfg)`: Sets up model, optimizer, scheduler, dataloaders, checkpoint directory
- `train()`: Main training loop with validation and checkpointing
- `evaluate(split)`: Computes classification accuracy on train/val/test splits
- `_setup()`: Initializes environment (seeds, CUDA settings, logging)

**Features**:
- Mixed precision training (AMP) support
- Automatic checkpointing to `model_checkpoints/{checkpoint_dir}/cfg{id}{seed_letter}/`
- WandB integration via `MetricsLogger`
- Gradient clipping and LR scheduling

---

### Evaluation

#### `evals.py`
**Purpose**: Orchestrates evaluation of trained models against neural data.

**Key Function**: `eval(cfg)`
**Pipeline**:
1. Load config (merges with training config if `load_model_from="checkpoint"`)
2. Load model and configure feature extractor
3. Load neural data (NSD/THINGS) and extract CNN activations
4. Optionally apply sparse random projection (SRP) for dimensionality reduction
5. Compute alignment scores (RSA or encoding)
6. Save results to CSV files in `logs/`

**Config Handling**:
- If `load_model_from="checkpoint"`: Loads training config from checkpoint and merges with eval config
- If `load_model_from="torchvision"`: Uses pretrained models directly

---

### Configuration

#### `config.py`
**Purpose**: Defines `ConfigDict` class for attribute-style config access.

**Usage**:
```python
cfg = ConfigDict({"mode": "train", "seed": 1})
print(cfg.mode)  # "train"
cfg.new_field = "value"  # Dynamic attribute assignment
```

#### `utils.py`
**Purpose**: Central utilities for the entire package.

**Key Functions**:
- **Config**: `load_config()`, `validate_config()`, `ConfigVerifier`
- **Optimization**: `setup_optimizer()`, `setup_scheduler()`
- **Logging**: `rprint()`, `MetricsLogger`, `save_results()`
- **File I/O**: `load_pickle()`, file locking for concurrent writes
- **Helpers**: `get_seed_letter()` (1→a, 2→b, 3→c), `calculate_cls_accuracy()`
- **Environment**: `get_env_var()`, `is_interactive_environment()`

**Config Validation** (`ConfigVerifier`):
- Validates `mode` ∈ {train, eval}
- Validates `dataset` ∈ {imagenet, tiny-imagenet, imagenet-mini-10/50/200}
- Validates `neural_dataset` ∈ {nsd, things, nsd_synthetic, cusack}
- Validates `analysis` ∈ {rsa, encoding_score}
- Validates `seed` ∈ {1, 2, 3}

---

## `analysis/` Subdirectory

Compute neural alignment scores between CNN representations and brain/behavioral data.

### `alignment.py`
**Purpose**: Main interface for computing neural alignment.

**Key Functions**:
- `compute_neural_alignment(cfg, activations, neural_data, ids)`: Routes to RSA or encoding based on `cfg.analysis`
- `prepare_data_for_alignment()`: Aligns activations with neural data by stimulus IDs, applies PCA reordering

**Supports**:
- RSA (Representational Similarity Analysis)
- Encoding models (ridge regression with LOO-CV)

### `rsa.py`
**Purpose**: Representational Similarity Analysis implementation.

**Key Functions**:
- `compute_rdm(representations, correlation="Pearson")`: Builds RDM (n_samples × n_samples) using Pearson/Spearman correlation
- `compute_rdm_correlation(rdm1, rdm2, correlation="Kendall")`: Correlates two RDMs (upper triangular parts)
- `compute_rsa_alignment(cfg, acts, neural_data)`: Per-layer RSA (no cross-validation). Always uses Pearson for RDMs and computes both Spearman and Kendall comparisons. Returns `score_spearman`/`score_kendall` per layer.
- `compute_rsa_kfold(cfg, acts, neural_data, n_folds=5, bootstrap=False)`: K-fold cross-validated RSA with unbiased layer selection. Always uses Pearson for RDMs and computes both Spearman and Kendall comparisons with independent layer selection per metric. Returns single-element list with dict containing:
  - `layer`: Spearman-selected layer (primary)
  - `score_spearman`, `score_kendall`: K-fold averaged scores per metric
  - `ci_low_{method}`, `ci_high_{method}`: Bootstrap CIs per metric
  - `fold_results_{method}`: List of `{"fold", "layer", "eval_score"}` per fold per metric
  - `layer_selection_scores_{method}`: List of `{"layer", "score"}` per metric
  - `bootstrap_scores_{method}`: List of floats (only when `bootstrap=True`)
  - (where `{method}` is `spearman` or `kendall`)
  - Saved to DB in normalized long format with `compare_method` column (see root CLAUDE.md)

**RDM Building**: Always Pearson. **RDM Comparison**: Always both Spearman and Kendall (no config needed).

### `encoding_score.py`
**Purpose**: GPU-accelerated ridge regression encoding models.

**Key Functions**:
- `ridge_regression_gpu(X_train, y_train, X_test, y_test, alphas)`: Ridge with LOO-CV for alpha selection
- `compute_encoding_alignment(cfg, acts, neural_data, ids)`: Fits ridge models per layer, returns Pearson R and R²
- `pearson_correlation(x, y)`: Vectorized Pearson correlation on GPU

**Output**: DataFrame with columns `[layer, region, r, r2, alpha, split]`

### `extract_representations.py`
**Purpose**: Extract intermediate layer activations from CNNs.

**Key Functions**:
- `extract_representations(model, dataloader, device, layers)`: Returns dict of `{layer_name: activations_tensor}`

### `sparse_random_projection.py`
**Purpose**: Apply SRP for fast dimensionality reduction before alignment.

**Benefits**: Reduces memory and computation for high-dimensional activations.

---

## `models/` Subdirectory

CNN architectures and model management utilities.

### `custom_model.py`
**Purpose**: Custom CNN implementations (AlexNet-like architectures).

**Key Classes**:
- `BaseCNN`: Abstract base class with shared functionality
  - Layer freezing via `trainable_layers` (e.g., `"11100"` = freeze fc layers)
  - Flexible nonlinearities (ReLU, GELU, LeakyReLU)
  - Configurable dropout, batch norm, pooling
- Subclasses: `AlexNetCustom`, `TinyAlexNetCustom`, etc.

**Architecture Control**:
- `conv_trainable` / `fc_trainable`: Binary strings to freeze/unfreeze layers
- Example: `conv_trainable="11111"` (all conv layers trainable), `fc_trainable="001"` (only last fc layer trainable)

### `standard_model.py`
**Purpose**: Wrappers for torchvision models (ResNet, AlexNet, VGG, DINOv2, CLIP).

**Key Classes**:
- `StandardAlexNet`, `StandardResNet`, etc.: Load pretrained models with layer freezing

### `ecnet.py`
**Purpose**: ECNet architecture (experimental).

### `utils.py`
**Purpose**: Model loading, checkpointing, and feature extraction.

**Key Functions**:
- `load_model(cfg, device, num_classes)`: Routes to custom or standard model based on `cfg.model_class`
- `save_checkpoint(dir, epoch, model, optimizer, metrics, cfg)`: Saves `.pth` and `config.json`
- `load_checkpoint(path, model, optimizer, device)`: Restores weights and optimizer state
- `setup_checkpoint_dir(cfg, model)`: Creates directory structure for checkpoints
- `configure_feature_extractor(cfg, model)`: Wraps model with `create_feature_extractor()` for layer extraction
- `get_activations(model, dataloader, device, apply_srp)`: Extracts activations for all samples

**Layer Naming**:
- Use `return_nodes` config to specify layers (e.g., `["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]`)

---

## `dataloaders/` Subdirectory

Data loading for training (ImageNet) and evaluation (neural datasets).

### `obj_cls.py`
**Purpose**: ImageNet and TinyImageNet dataloaders with PCA label support.

**Key Functions**:
- `get_obj_cls_loader(cfg)`: Returns datasets and dataloaders for train/val/test splits
- `get_transform(ds_stats, data_augment, image_size)`: Returns torchvision transforms

**PCA Label Support**:
- If `cfg.pca_labels=True`: Loads coarse-grained labels from `datasets/obj_cls/imagenet/pca_labels_{global/hierarchical}/pca{n_classes}_seed{seed}.pkl`
- Supports 2–1000 classes via PCA-based label coarsening

**Datasets**:
- `imagenet`: Full ImageNet (1000 classes)
- `tiny-imagenet`: TinyImageNet (200 classes, 64×64 images)
- `imagenet-mini-{10,50,200}`: Subsets of ImageNet

### `neural.py`
**Purpose**: Neural datasets (fMRI and behavioral).

**Key Functions**:
- `get_neural_loader(cfg)`: Returns neural data dict and dataloader based on `cfg.neural_dataset`
- `load_nsd_data(cfg)`: Loads NSD fMRI responses for specific subject and brain region
- `load_things_data(cfg)`: Loads THINGS behavioral embeddings

**Datasets**:
- **NSD** (Natural Scenes Dataset): fMRI responses to images
  - Regions: EVC, IT, V1-V4, LOC, etc.
  - Subjects: 1-8
- **THINGS**: Behavioral similarity judgments (embedding space)

**Data Structure**:
- Returns `(neural_data, dataloader)` where:
  - `neural_data`: Dict mapping `{layer_name: target_tensor}`
  - `dataloader`: PyTorch DataLoader for stimulus images

---

## Key Workflows

### Training a Model
1. `python -m visreps.run --mode train --config configs/train/base.json`
2. `run.main()` → `Trainer(cfg)` → `trainer.train()`
3. Model checkpoints saved to `model_checkpoints/{checkpoint_dir}/cfg{id}{seed_letter}/`

### Evaluating a Model
1. `python -m visreps.run --mode eval --override seed=1 analysis=rsa`
2. `run.main()` → `evals.eval(cfg)`
3. Load model → Extract activations → Compute alignment → Save to `logs/`

### Adding a New Analysis Method
1. Implement in `visreps/analysis/` (e.g., `new_method.py`)
2. Add to `alignment.py::compute_neural_alignment()` dispatch logic
3. Update `ConfigVerifier` in `utils.py` to accept new `analysis` value

### Adding a New Model Architecture
1. Define in `visreps/models/custom_model.py` or `standard_model.py`
2. Update `models/utils.py::load_model()` to handle new `model_class` value
3. Ensure layer naming is consistent for feature extraction

---

## Important Patterns

### Config Merging
- Base config → Grid config → CLI overrides (highest priority)
- Training configs merged with eval configs when loading from checkpoints

### Layer Extraction
- Use `return_nodes` in config to specify which layers to extract
- Layer names depend on model architecture (see `models/utils.py::configure_feature_extractor()`)

### File Locking
- `utils.save_results()` uses file locks for concurrent writes (grid search jobs)

### Seed Management
- Seeds 1, 2, 3 map to letters a, b, c via `get_seed_letter()`
- Used in checkpoint directory naming

### Sparse Random Projection
- Set `apply_srp=True` in config to reduce dimensionality before alignment
- Speeds up encoding/RSA for high-dimensional layers

---

## Environment Variables (from `.env`)

Required for data loading:
- `IMAGENET_DATA_DIR`: Path to ImageNet images
- `NSD_DATA_DIR`: Path to NSD fMRI data
- Other neural dataset paths as needed

---

## Testing

Run tests from repo root:
```bash
pytest tests/
pytest tests/test_training_pipeline.py -v
```

Test files mirror package structure:
- `tests/test_training_pipeline.py`: Tests `Trainer` class
- `tests/test_model_setup.py`: Tests model loading
- `tests/test_utils.py`: Tests utility functions

---

## Common Gotchas

1. **Layer Names**: Ensure `return_nodes` match actual model layer names (use `print(model)` to verify)
2. **PCA Labels**: Must exist in `datasets/obj_cls/imagenet/pca_labels_*/` before training
3. **Checkpoint Paths**: Use `checkpoint_dir`, `cfg_id`, and `seed` to locate saved models
4. **Config Validation**: Invalid config values will raise errors from `ConfigVerifier`
5. **Neural Data Alignment**: Stimulus IDs must match between CNN activations and neural data

---

## Further Reading

See root-level `CLAUDE.md` for:
- High-level project overview
- Common CLI commands
- Grid search workflows
- External data paths
