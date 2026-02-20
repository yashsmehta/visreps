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
**Purpose**: Unified evaluation of trained models against neural data. One forward pass per seed handles all subjects and regions internally.

**Key Function**: `eval(cfg)`
Accepts list-valued `cfg.subject_idx` and `cfg.region`. Routes to dataset-specific paths:

**NSD/TVSD path** (multi-subject):
1. Load config (merges with training config if `load_model_from="checkpoint"`)
2. Load model and configure feature extractor
3. Load all neural data via `load_all_nsd_data` or `load_all_tvsd_data` (one call for all subjects/regions)
4. Extract activations once with SRP
5. Dispatch to `_eval_rsa` or `_eval_encoding` helper
6. Save per-subject per-region results to `results.db`

**THINGS-behavior path** (80/20 concept-level train/test, no subjects/regions):
1. Load model, extract activations
2. Merge train/test images, average activations per concept via `prepare_concept_alignment`
3. Split 80/20 at concept level (seed=42), call `compute_traintest_alignment` (train/test RSA)
4. Save results to `results.db`

**Internal helpers**:
- `_eval_rsa(cfg, model, acts, ids, all_data, subjects, regions, dev, verbose)`: Two-phase RSA. Phase 1: per-(subject, region) layer selection using SRP activations. Phase 2: re-extract unique best layers without SRP, compute per-subject test RDMs and optional bootstrap CIs.
- `_eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose)`: Per-(subject, region) encoding score using SRP activations throughout (no re-extraction).
- `_load_cfg(cfg)`: Merges runtime config with training config from checkpoint.
- `_listify(val)`: Normalizes int/str/ListConfig to plain Python list.

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
- Validates `mode` in {train, eval}
- Validates `dataset` in {imagenet, tiny-imagenet, imagenet-mini-10/50/200}
- Validates `neural_dataset` in {nsd, things-behavior, tvsd}
- Validates `analysis` in {rsa, encoding_score}
- Validates `compare_method` in {spearman, kendall}
- Validates `seed` in {1, 2, 3}
- For NSD/TVSD: normalizes `subject_idx` and `region` to lists, validates each element
- For things-behavior: sets `region` and `subject_idx` to "N/A"
- For encoding_score: overrides `compare_method` to "pearson"

---

## `analysis/` Subdirectory

Compute neural alignment scores between CNN representations and brain/behavioral data.

### `alignment.py`
**Purpose**: Main interface for computing neural alignment. Provides data preparation and dispatch for train/test evaluation.

**Key Functions**:
- `_align_stimulus_level(acts_raw, targets, keys)`: Aligns activations with neural targets by stimulus ID. Returns (acts, neural, matched_ids).
- `prepare_traintest_alignment(cfg, acts_raw, neural_data_raw, keys)`: Aligns activations with train/test neural data for NSD/TVSD (stimulus-level) or THINGS (concept-level). Returns two `AlignmentData` objects.
- `compute_traintest_alignment(cfg, train, test, verbose, re_extract_fn)`: Dispatches to `compute_rsa` or `compute_encoding_score` based on `cfg.analysis`.
- `prepare_concept_alignment(cfg, acts_raw, neural_data_raw, keys)`: Merges train/test THINGS images, averages activations per concept. Returns single `AlignmentData` with all ~1,854 concepts.

**Data class**: `AlignmentData` bundles activations, neural data, stimulus_ids, and optional concept_image_ids.

### `rsa.py`
**Purpose**: Representational Similarity Analysis implementation.

**Key Functions**:
- `compute_rdm(representations, correlation="Pearson")`: Builds RDM (n_samples x n_samples) using Pearson/Spearman correlation.
- `compute_rdm_correlation(rdm1, rdm2, correlation="Kendall")`: Correlates two RDMs (upper triangular parts). Supports Pearson, Spearman, and Kendall (tau-a).
- `compute_rsa(cfg, selection, evaluation, ...)`: Train/test RSA for all datasets. Subsamples train stimuli for layer selection, evaluates best layer on test set with optional re-extraction without SRP and bootstrap CIs. Returns single-element list with result dict.

**RDM Building**: Always Pearson. **RDM Comparison**: Single metric from `cfg.compare_method` (`"spearman"` or `"kendall"`).

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
- `get_activations(model, dataloader, device)`: Extracts activations for all samples (always applies SRP)
- `extract_single_layer(model, dataloader, device, layer_name, stimulus_ids)`: Re-extracts one layer without SRP for exact RDMs

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
**Purpose**: Neural datasets (fMRI, electrophysiology, and behavioral).

**Key Functions**:
- `get_neural_loader(cfg)`: Returns `(targets, dataloader)` for a single subject/region. Used by the THINGS path.
- `load_nsd_data(cfg)`: Loads NSD fMRI responses for a single subject and brain region. Returns `(targets, stimuli)`.
- `load_all_nsd_data(cfg, subjects=None, regions=None)`: Loads NSD fMRI for all requested subjects and regions at once. Returns dict with keys: `"regions"`, `"subjects"`, `"neural"` (nested `{region: {subj: {"train": ..., "test": ...}}}`), `"stimuli"` (lazy HDF5 dict), `"shared_test_ids"`. Used by unified eval for NSD.
- `load_tvsd_data(cfg)`: Loads TVSD macaque MUA responses for a single subject/region.
- `load_all_tvsd_data(cfg, subjects=None, regions=None)`: Loads TVSD for all requested subjects and regions. Same return structure as `load_all_nsd_data` but with image paths instead of lazy HDF5. Used by unified eval for TVSD.
- `load_things_data()`: Loads THINGS behavioral dataset with within-concept train/test image split from cached pickle.
- `_make_loader(stimuli, transform, batch, workers)`: Creates a DataLoader from a stimuli dict.

**Datasets**:
- **NSD** (Natural Scenes Dataset): fMRI responses to images
  - Regions: "early visual stream", "ventral visual stream"
  - Subjects: 0-7
- **TVSD**: Macaque electrophysiology (MUA)
  - Regions: "V1", "V4", "IT"
  - Subjects: 0 (monkey F), 1 (monkey N)
- **THINGS**: Behavioral similarity judgments (embedding space)

**Data Structure**:
- Single-subject loaders return `(targets, stimuli)` where targets is `{"train": {sid: response}, "test": {sid: response}}`
- Multi-subject loaders return a dict with nested `neural[region][subj]` structure and shared stimuli/test IDs

---

## Key Workflows

### Training a Model
1. `python -m visreps.run --mode train --config configs/train/base.json`
2. `run.main()` -> `Trainer(cfg)` -> `trainer.train()`
3. Model checkpoints saved to `model_checkpoints/{checkpoint_dir}/cfg{id}{seed_letter}/`

### Evaluating a Model
1. `python -m visreps.run --mode eval --override seed=1 analysis=rsa`
2. `run.main()` -> `evals.eval(cfg)`
3. For NSD/TVSD: Load model -> Load all neural data (`load_all_nsd_data`/`load_all_tvsd_data`) -> Extract activations once -> `_eval_rsa` or `_eval_encoding` iterates over (subject, region) pairs -> Save per-subject per-region results to `results.db`
4. For THINGS: Load model -> Extract activations -> `prepare_concept_alignment` (merge + concept-average) -> 80/20 concept split -> `compute_traintest_alignment` (train/test RSA) -> Save to `results.db`

### Adding a New Analysis Method
1. Implement in `visreps/analysis/` (e.g., `new_method.py`)
2. Add to `alignment.py::compute_traintest_alignment()` dispatch logic
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
- SRP is always applied during activation extraction (`get_activations`) to keep memory bounded
- For RSA, the best layer is re-extracted without SRP via `extract_single_layer` for exact test RDMs
- Encoding score uses SRP throughout (features are ridge regression inputs)

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
4. **Config Validation**: Invalid config values will raise errors from `ConfigVerifier`. For eval, `subject_idx` and `region` are normalized to lists.
5. **Neural Data Alignment**: Stimulus IDs must match between CNN activations and neural data
6. **THINGS**: THINGS-behavior uses 80/20 concept-level train/test split (fixed seed=42); `region` and `subject_idx` are set to "N/A"

---

## Further Reading

See root-level `CLAUDE.md` for:
- High-level project overview
- Common CLI commands
- Grid search workflows
- External data paths
