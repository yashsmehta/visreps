# CLAUDE.md
I want simple, intuitive and easy to understand code. If the user is asking for a complex code change / ambiguous query (which could be interpretted in multiple ways), first you should read the relevant files, and confirm with the user that what you are thinking is indeed what the user meant before going ahead with code changes and implementation.

**Always activate the virtual environment first:**
```bash
source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate
```

## Project Overview

Investigates whether **fine-grained category supervision is necessary for brain-model alignment**. Trains CNNs on ImageNet with varying label granularity (2-1000 classes via PCA-based coarse labels), then evaluates alignment with human visual cortex (NSD fMRI) or behavioral data (THINGS).

## Repository Structure

```
visreps/                   # Main package
├── run.py                 # Entry point (--mode train|eval)
├── trainer.py             # Training loop
├── evals.py               # Evaluation orchestration
├── utils.py               # Config validation, logging, optimization
├── analysis/              # RSA, encoding_score, SRP
├── models/                # CustomCNN, standard torchvision wrappers
└── dataloaders/           # obj_cls.py (ImageNet), neural.py (NSD/THINGS)

configs/                   # JSON configs (train/, eval/, grids/)
scripts/
├── slurm/                 # train_scheduler.py, eval_scheduler.py
├── runners/               # Local experiment runners
├── coarsegrain/           # PCA label generation
└── extract_representations/  # Feature extraction from pretrained models

pca_labels/                # Generated coarse labels (pca_labels_{model}/n_classes_{n}.csv)
model_checkpoints/         # Saved models: {checkpoint_dir}/cfg{n_classes}{seed_letter}/
logs/                      # Evaluation results (CSV)
```

## Training (`--mode train`)

```bash
python -m visreps.run --mode train --override pca_labels=true pca_n_classes=32 seed=1
python scripts/slurm/train_scheduler.py  # SLURM grid search
```

**Key config options:**
- `dataset`: "imagenet", "tiny-imagenet", "imagenet-mini-{10,50,200}"
- `pca_labels`: true/false (use coarse labels)
- `pca_n_classes`: 2, 4, 8, 16, 32, 64 (must be power of 2)
- `pca_labels_folder`: "pca_labels_alexnet", "pca_labels_dino", etc.
- `model_class`: "custom_model" or "standard_model"

**Checkpoint naming:** `model_checkpoints/{checkpoint_dir}/cfg{n_classes}{seed_letter}/`
- seed_letter: 1→a, 2→b, 3→c

## Evaluation (`--mode eval`)

```bash
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=rsa neural_dataset=nsd
python scripts/slurm/eval_scheduler.py  # SLURM grid search
```

**Key config options:**
- `load_model_from`: "checkpoint" or "torchvision"
- `cfg_id`: must match `pca_n_classes` from training (or 1000 for standard)
- `neural_dataset`: "nsd", "things", "nsd_synthetic", "cusack"
- `analysis`: "rsa" or "encoding_score"
- `region`: "early visual stream", "ventral visual stream" (NSD only)
- `subject_idx`: 0-7 (NSD only)
- `return_nodes`: ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
- `apply_srp`: true (Sparse Random Projection for speed)

**Analysis methods:**
- **RSA**: Correlates model and neural Representational Similarity Matrices
- **Encoding**: Ridge regression (himalaya GPU) predicting neural responses from activations

## PCA Labels Generation

```bash
python scripts/extract_representations/alexnet_representations.py  # Extract features
python scripts/coarsegrain/compute_eigenvectors.py                 # Compute PCs
python scripts/coarsegrain/make_pca_labels.py                      # Generate labels
```

Creates labels by projecting features onto PCs and applying median splits → 2^N classes.

## Configuration

- Base config → nested config merged (`custom_model`/`standard_model` or `checkpoint`/`torchvision`) → CLI `--override`
- Grid configs in `configs/grids/` define parameter arrays for sweeps
- `ConfigVerifier` validates: mode, dataset, neural_dataset, analysis, seed (1-3)

## Environment Variables (`.env`)

```
IMAGENET_DATA_DIR=/path/to/imagenet/train
IMAGENET_LOCAL_DIR=/path/to/imagenet          # Contains folder_labels.json
TINY_IMAGENET_DATA_DIR=/path/to/tiny-imagenet-200
NSD_DATA_DIR=/path/to/nsd/processed
```

## Models

- **CustomCNN / TinyCustomCNN**: AlexNet-style, configurable layer freezing via `conv_trainable`/`fc_trainable` binary strings
- **Standard models**: AlexNet, VGG16, ResNet18, ResNet50, ViTBase (torchvision wrappers)

## Key Gotchas

1. PCA labels must exist in `pca_labels/{folder}/n_classes_{n}.csv` before training
2. `cfg_id` in eval must match `pca_n_classes` from training
3. THINGS dataset ignores `region` and `subject_idx`
4. Layer names: CNNs use `conv1-5`, `fc1-2`; ViT uses `block1-12`, `head`
