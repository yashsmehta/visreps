# CLAUDE.md

Write simple, intuitive, easy-to-understand code. For complex or ambiguous requests, read the relevant files first and confirm intent with the user before implementing.

## Voice Dictation

Instructions are mostly voice-dictated and may contain transcription errors. Infer intent from context, state any assumptions before proceeding, and ask only when genuinely ambiguous.

## Environment

**CRITICAL: Always activate the venv before running any Python command.** Prefix every Python invocation with `source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate &&`. The system Python does not have the required packages (torch, etc.).

**All scripts must be run from the project root** (`/home/ymehta3/research/VisionAI/visreps/`), not from subdirectories. The dataloaders use relative paths (e.g. `datasets/obj_cls/imagenet/folder_labels.json`) that resolve from root.

## Project Overview

Investigates whether **fine-grained category supervision is necessary for brain-model alignment**. Trains CNNs on ImageNet with varying label granularity (2-1000 classes via PCA-based coarse labels), then evaluates alignment with human visual cortex (NSD fMRI), behavioral data (THINGS), macaque electrophysiology (TVSD), or infant fMRI (Cusack).

## Repository Structure

```
visreps/                   # Main package
├── run.py                 # Entry point (--mode train|eval)
├── trainer.py             # Training loop
├── evals.py               # Evaluation orchestration
├── utils.py               # Config validation, logging, optimization
├── analysis/              # RSA, encoding_score, SRP
├── models/                # CustomCNN, standard torchvision wrappers
└── dataloaders/           # obj_cls.py (ImageNet), neural.py (NSD/THINGS/TVSD/Cusack)

configs/                   # JSON configs (train/, eval/, grids/)
scripts/
├── slurm/                 # Slurm schedulers (Rockfish cluster only, not used here)
├── runners/               # Local experiment runners (use these on this machine)
├── coarsegrain/           # PCA label generation
└── extract_representations/  # Feature extraction from pretrained models

pca_labels/                # Generated coarse labels (pca_labels_{model}/n_classes_{n}.csv)
model_checkpoints/         # Saved models: {checkpoint_dir}/cfg{n_classes}{seed_letter}/
logs/                      # Evaluation results (SQLite DB + legacy CSVs)
```

## Training (`--mode train`)

```bash
python -m visreps.run --mode train --override pca_labels=true pca_n_classes=32 seed=1
python scripts/runners/train_runner.py --grid configs/grids/train_default.json  # Grid sweep (local)
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
python scripts/runners/eval_runner.py --grid configs/grids/eval_default.json  # Grid sweep (local)
```

**Note:** `scripts/slurm/eval_scheduler.py` is for the Rockfish Slurm cluster only. This machine is a local lab GPU cluster — always use `scripts/runners/eval_runner.py` instead, which loads grid configs from `configs/grids/` via `--grid`.

**Key config options:**
- `load_model_from`: "checkpoint" or "torchvision"
- `cfg_id`: must match `pca_n_classes` from training (or 1000 for standard)
- `neural_dataset`: "nsd", "things", "nsd_synthetic", "cusack", "tvsd"
- `analysis`: "rsa" or "encoding_score"
- `region`: NSD: "early visual stream", "ventral visual stream"; TVSD: "V1", "V4", "IT"; Cusack: region string
- `subject_idx`: NSD: 0-7; TVSD: 0 (monkey F), 1 (monkey N); THINGS: ignored
- `return_nodes`: ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
- `apply_srp`: true (Sparse Random Projection for speed)
- `bootstrap`: true/false (compute bootstrap CIs on last fold of k-fold RSA)

**Analysis methods:**
- **RSA**: K-fold cross-validated RSA with unbiased layer selection. Always builds RDMs with Pearson correlation and computes both Spearman and Kendall-tau comparisons (independent layer selection per metric). Reported `layer` is the Spearman-selected layer.
- **Encoding**: Ridge regression (himalaya GPU) predicting neural responses from activations

## Results Database

Eval results are stored in `results.db` (SQLite).

**Tables** (normalized "long" format — each comparison metric gets its own row via `compare_method`):
- `results` — one row per (run, compare_method, layer). Columns: `run_id`, `compare_method`, `layer`, `score`, `ci_low`, `ci_high`, `analysis`, `seed`, `epoch`, `region`, `subject_idx`, `neural_dataset`, `cfg_id`, `pca_labels`, `pca_n_classes`, `pca_labels_folder`, `model_name`, `checkpoint_dir`, `reconstruct_from_pcs`, `pca_k`. Deduped via `UNIQUE(run_id, compare_method, layer)`.
- `run_configs` — full config JSON per `run_id`. Use `JOIN` to recover any training/infra parameter.
- `fold_results` — per-fold eval score + selected layer (10 rows per run for 5-fold CV × 2 metrics). Columns: `run_id`, `compare_method`, `fold`, `layer`, `eval_score`. Deduped via `UNIQUE(run_id, compare_method, fold)`.
- `layer_selection_scores` — per-layer mean selection score across folds (~14 rows per metric per run). Columns: `run_id`, `compare_method`, `layer`, `score`. Deduped via `UNIQUE(run_id, compare_method, layer)`.
- `bootstrap_distributions` — raw bootstrap scores as JSON array (1 row per metric per run, only when `bootstrap=True`). Columns: `run_id`, `compare_method`, `scores`.

`compare_method` is `"spearman"` or `"kendall"`. `run_id` = SHA256[:12] of experiment identity fields. Re-running the same eval replaces old results (INSERT OR REPLACE).

**Quick query:**
```python
pd.read_sql("SELECT * FROM results WHERE region='ventral visual stream' AND compare_method='spearman'", sqlite3.connect("results.db"))
```

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
BONNER_DATASETS_HOME=~/.cache/bonner-datasets  # TVSD uses THINGS images from here
```

## Models

- **CustomCNN / TinyCustomCNN**: AlexNet-style, configurable layer freezing via `conv_trainable`/`fc_trainable` binary strings
- **Standard models**: AlexNet, VGG16, ResNet18, ResNet50, ViTBase (torchvision wrappers)

## Key Gotchas

1. **Run from project root** — relative paths in dataloaders break otherwise
2. PCA labels must exist in `pca_labels/{folder}/n_classes_{n}.csv` before training
3. `cfg_id` in eval must match `pca_n_classes` from training
4. THINGS dataset ignores `region` and `subject_idx`
5. TVSD requires pre-cached pickle at `datasets/neural/tvsd/fmri_responses.pkl` (generate with `python scripts/cache_tvsd_data.py`); uses THINGS stimulus images via `BONNER_DATASETS_HOME`
6. Layer names: CNNs use `conv1-5`, `fc1-2`; ViT uses `block1-12`, `head`
7. Checkpoints at `/data/ymehta3/default/` (1000-way) and `/data/ymehta3/alexnet_pca/` (coarse)
