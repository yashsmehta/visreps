# CLAUDE.md

Write simple, intuitive, easy-to-understand code. For complex or ambiguous requests, read the relevant files first and confirm intent with the user before implementing.

## Voice Dictation

Instructions are mostly voice-dictated and may contain transcription errors. Infer intent from context, state any assumptions before proceeding, and ask only when genuinely ambiguous.

## Environment

**Hardware:** 1× NVIDIA RTX 4090 (24 GB VRAM), 32 CPU cores, 125 GB RAM. Encoding score (himalaya ridge SVD) needs ~8 GB free VRAM for NSD-sized data.

**CRITICAL: Always activate the venv before running any Python command.** Prefix every Python invocation with `source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate &&`. The system Python does not have the required packages (torch, etc.).

**Always load `.env` before importing packages** that depend on environment variables (e.g., `BONNER_DATASETS_HOME`). Use `source .env` or `python-dotenv`. Without this, datasets re-download to `~/.cache` instead of the configured data directory.

**All scripts must be run from the project root** (`/home/ymehta3/research/VisionAI/visreps/`), not from subdirectories. The dataloaders use relative paths (e.g. `datasets/obj_cls/imagenet/folder_labels.json`) that resolve from root.

## Project Overview

Investigates whether **fine-grained category supervision is necessary for brain-model alignment**. Trains CNNs on ImageNet with varying label granularity (2-1000 classes via PCA-based coarse labels), then evaluates alignment with human visual cortex (NSD fMRI), behavioral data (THINGS), or macaque electrophysiology (TVSD).

## Repository Structure

```
visreps/                   # Main package
├── run.py                 # Entry point (--mode train|eval)
├── trainer.py             # Training loop
├── evals.py               # Evaluation orchestration
├── utils.py               # Config validation, logging, optimization
├── analysis/              # RSA, encoding_score, SRP
├── models/                # CustomCNN, standard torchvision wrappers
└── dataloaders/           # obj_cls.py (ImageNet), neural.py (NSD/THINGS/TVSD)

configs/                   # JSON configs (train/, eval/, grids/)
scripts/
├── slurm/                 # Slurm schedulers (Rockfish cluster only, not used here)
├── runners/               # Local experiment runners (use these on this machine)
├── coarsegrain/           # PCA label generation
└── extract_representations/  # Feature extraction from pretrained models

plotters/                  # Publication-quality plotting scripts
├── plotter_utils.py       # Shared helpers (DB queries, data aggregation)
├── plot_helpers.py        # Shared coarseness plotting logic (bars, per-subject)
├── plot_architectures.py  # Architecture comparison plots
├── nsd/                   # NSD coarseness scripts + figures
│   ├── plot_coarseness.py           # Early + ventral visual streams
│   ├── plot_coarseness_finegrained.py  # 6 ROIs (V1-PPA), 2×4 layout
│   └── figures/
├── nsd_synthetic/         # NSD-synthetic coarseness scripts + figures
│   ├── plot_coarseness.py           # 6 ROIs, 2×4 layout
│   └── figures/
├── tvsd/                  # TVSD coarseness scripts + figures
│   ├── plot_coarseness.py           # V1, V4, IT
│   └── figures/
└── things/                # THINGS coarseness scripts + figures
    ├── plot_coarseness.py           # Single panel, no per-subject
    └── figures/

experiments/               # Self-contained analyses (each dir has own data + scripts)
├── stimulus_robustness/   # RSA under stimulus subsampling (analysis.py + plot.py)
├── stimulus_sensitivity/  # k-fold CV RSA fluctuation (plot.py)
├── neurips_2025/          # NeurIPS submission figures
├── coarse_grain_benefits/ # Few-shot, robustness, curriculum, linear probe
├── representation_analysis/
└── ...

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

There is a single unified `eval` mode (no separate aggregate mode). For NSD/TVSD, one forward pass per seed handles all subjects and regions internally. Per-subject per-region results are saved individually.

```bash
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=rsa neural_dataset=nsd
python scripts/runners/eval_runner.py --grid configs/grids/eval_default.json  # Grid sweep (local)
```

**Note:** `scripts/slurm/eval_scheduler.py` is for the Rockfish Slurm cluster only. This machine is a local lab GPU cluster — always use `scripts/runners/eval_runner.py` instead, which loads grid configs from `configs/grids/` via `--grid`.

**Key config options:**
- `load_model_from`: "checkpoint" or "torchvision"
- `cfg_id`: must match `pca_n_classes` from training (or 1000 for standard)
- `neural_dataset`: "nsd", "things-behavior", "tvsd"
- `analysis`: "rsa" or "encoding_score"
- `region`: list or string. NSD: `["early visual stream", "ventral visual stream"]`; TVSD: `["V1", "V4", "IT"]`. Scalars are normalized to lists by `ConfigVerifier`.
- `subject_idx`: list or int. NSD: `[0,1,2,3,4,5,6,7]`; TVSD: `[0, 1]`. Scalars are normalized to lists. THINGS: ignored (set to "N/A").
- `return_nodes`: ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
- `compare_method`: "spearman" (default) or "kendall" — which RDM comparison metric to use
- `bootstrap`: true/false (compute bootstrap CIs)

**Unified eval behavior:**
- **NSD/TVSD**: Loads all neural data for requested subjects/regions via `load_all_nsd_data`/`load_all_tvsd_data`. Extracts model activations once, then iterates over all (subject, region) pairs to compute and save per-subject per-region results.
- **THINGS-behavior**: Uses 80/20 concept-level train/test split (no subjects or regions). Merges train/test images, averages activations per concept, splits 20% for layer selection and 80% for evaluation + bootstrap.

**Grid config patterns:**
- NSD grid configs typically omit `subject_idx` and `region` (defaults come from `configs/eval/base.json`).
- TVSD grid configs use a nested list pattern for `subject_idx` (e.g., `"subject_idx": [[0, 1]]`) so that the Cartesian product yields one list per seed rather than expanding to individual subjects.

### SRP, Datasets, and Analysis Pipelines

**Sparse Random Projection (SRP):** All layer activations are projected to k=4096 dims (`SparseRandomProjection`, cached in `model_checkpoints/srp_cache/`) to keep memory bounded. RSA re-extracts the best layer *without* SRP for exact test RDMs; encoding score uses SRP throughout.

**Datasets:** NSD (~9k train / ~1k test, 8 subjects, early/ventral visual stream), TVSD (~22k train / 100 test, 2 monkeys, V1/V4/IT), THINGS (~1,854 concepts, 80/20 concept-level train/test split).

**RSA — NSD/TVSD:** Per subject: select best layer on train (subsample 1,000 stimuli, build Pearson RDMs, compare via Spearman/Kendall) → re-extract best layer without SRP → score on test RDMs → 1,000-iteration bootstrap (90% subsample) for 95% CIs.

**RSA — THINGS:** Fixed 80/20 concept-level split (seed=42). 20% (~370 concepts) for layer selection, 80% (~1,480 concepts) for evaluation. Re-extract best layer without SRP, concept-average for eval set. Bootstrap 1,000 iterations (90% subsample) on eval set for 95% CIs.

**Encoding — NSD/TVSD only** (Pearson r, not applicable to THINGS): Per subject: select best layer via 80/20 fit/val split with `RidgeCV(cv=5)` → refit on full train → predict test → mean Pearson r across voxels → 1,000-iteration bootstrap on cached predictions for 95% CIs. Uses SRP throughout, z-normalization with fit-only stats during selection.

## Results Database

Eval results are stored in `results.db` (SQLite).

**Tables** (one row per run per metric — `compare_method` is part of the `run_id` hash):
- `results` — one row per (run, compare_method, layer). Columns: `run_id`, `compare_method`, `layer`, `score`, `ci_low`, `ci_high`, `analysis`, `seed`, `epoch`, `region`, `subject_idx`, `neural_dataset`, `cfg_id`, `pca_labels`, `pca_n_classes`, `pca_labels_folder`, `model_name`, `checkpoint_dir`, `reconstruct_from_pcs`, `pca_k`. Deduped via `UNIQUE(run_id, compare_method, layer)`.
- `run_configs` — full config JSON per `run_id`. Use `JOIN` to recover any training/infra parameter.
- `layer_selection_scores` — per-layer selection score (~7 rows per run). Columns: `run_id`, `compare_method`, `layer`, `score`. Deduped via `UNIQUE(run_id, compare_method, layer)`.
- `bootstrap_distributions` — raw bootstrap scores as JSON array (1 row per run, only when `bootstrap=True`). Columns: `run_id`, `compare_method`, `scores`.

`compare_method` is `"spearman"` or `"kendall"` (set via `cfg.compare_method`). `run_id` = SHA256[:12] of experiment identity fields (includes `compare_method`). Re-running the same eval replaces old results (INSERT OR REPLACE).

**NEVER modify `results.db` directly** (no DELETE, UPDATE, DROP, or any write operations). The database is the single source of truth for all evaluation results. If cleanup or modification is needed, describe the operation and provide the SQL command to the user for them to run manually.

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
- `ConfigVerifier` validates: mode, dataset, neural_dataset, analysis, compare_method, seed (1-3)

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

## Evaluation Pipeline

Use `scripts/runners/eval_runner.py` with JSON grid configs (`configs/grids/`) for running evaluations — **not** `scripts/slurm/eval_scheduler.py` (which is Slurm-only with hardcoded param grids). Always check the JSON configs in `configs/grids/` before running experiments.

## Model Checkpoints

Always verify checkpoint paths exist with `ls` before referencing them in code. Common locations:
- `/data/ymehta3/default/` — 1000-way (standard ImageNet)
- `/data/ymehta3/alexnet_pca/` — coarse-grained (PCA labels)

Do **not** guess checkpoint paths.

## Plotting & Visualization

Coarseness plots are organized per-dataset: `plotters/{nsd,nsd_synthetic,tvsd,things}/` each contain a `plot_coarseness.py` script and a `figures/` output directory. Shared plotting logic lives in `plotters/plot_helpers.py`. Architecture comparison plots use `plotters/plot_architectures.py`. Stimulus robustness and sensitivity analyses are self-contained in `experiments/stimulus_robustness/` and `experiments/stimulus_sensitivity/`.

```bash
python plotters/nsd/plot_coarseness.py --folder pca_labels_alexnet
python plotters/nsd/plot_coarseness_finegrained.py --folder pca_labels_clip
python plotters/tvsd/plot_coarseness.py --folder pca_labels_alexnet
python plotters/things/plot_coarseness.py --folder pca_labels_alexnet
```

When asked for a single plot or specific layout, implement exactly that — do not add extra subplots or change the layout without asking. Prefer minimal, publication-quality single-panel figures unless told otherwise.

## Key Gotchas

1. **Run from project root** — relative paths in dataloaders break otherwise
2. PCA labels must exist in `pca_labels/{folder}/n_classes_{n}.csv` before training
3. `cfg_id` in eval must match `pca_n_classes` from training
4. `things-behavior` ignores `region` and `subject_idx` (set to "N/A"); uses 80/20 concept-level train/test split with fixed seed=42
5. TVSD and `things-behavior` require pre-cached pickles (generate with `python scripts/preprocess_data/preprocess_tvsd.py` and `python scripts/preprocess_data/preprocess_things.py`); both use THINGS stimulus images via `BONNER_DATASETS_HOME`
6. Layer names: CNNs use `conv1-5`, `fc1-2`; ViT uses `block1-12`, `head`
7. Checkpoints at `/data/ymehta3/default/` (1000-way) and `/data/ymehta3/alexnet_pca/` (coarse)
8. **SRP (k=4096) is always applied during `get_activations()`**. For RSA, the best layer is re-extracted without SRP via `extract_single_layer` for exact test RDMs. Encoding score uses SRP throughout. See "Sparse Random Projection" section above for details.
