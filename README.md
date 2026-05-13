<div align="center">

# visreps

### An extremely coarse feedback signal is sufficient for learning<br/>human-aligned visual representations

<p>
<a href="https://arxiv.org/abs/2605.05556"><img src="https://img.shields.io/badge/arXiv-2605.05556-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<img src="https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge" alt="Under Review">
<img src="https://img.shields.io/badge/VSS%202026-Oral%20Talk-8A2BE2?style=for-the-badge" alt="VSS 2026 Oral">
</p>

<p>
<a href="https://docs.python.org/3/whatsnew/3.11.html"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python" alt="Python Version"></a>
<a href="https://www.pytorch.com/"><img src="https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch&labelColor=gray" alt="PyTorch"></a>
<a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative" alt="License"></a>
</p>

</div>

---

## Overview

It is widely assumed that **fine-grained category supervision** (e.g., 1000-way ImageNet) is what drives the human-like representations learned by deep vision models.

**We show this is wrong.** Networks trained with as few as **2–8 broad categories** match — and on perceptual similarity, *exceed* — the brain-alignment of 1000-class models and large self-supervised systems.

We construct progressively coarser label sets by PCA-partitioning pretrained embeddings, retrain CNNs and transformers from scratch at each granularity, and evaluate alignment against:

| Modality | Dataset | Regions / Signal |
|---|---|---|
| Human fMRI | **NSD** | Early & ventral visual streams |
| Macaque electrophysiology | **TVSD** | V1, V4, IT |
| Human behavior | **THINGS** | Similarity judgments |

---

## What this repository provides

End-to-end machinery for systematically varying label granularity and measuring its effect on brain/behavior alignment.

| | Capability |
|---|---|
| **Coarse-label generation** | PCA on pretrained features (AlexNet, CLIP, DINOv3, supervised ViT) → median-split into 2ⁿ hierarchical classes. Pixel-PCA labels included as a learned-feature-free control. |
| **Training** | CustomCNN, ResNet, ConvNeXt, ViT on ImageNet at any granularity from 2 to 1,000, with AMP, schedulers, and seed-tagged checkpoints. |
| **Brain alignment** | RSA (Spearman/Kendall on Pearson RDMs) and encoding scores (RidgeCV) against NSD, TVSD, and THINGS, with 1,000-iter bootstrap 95% CIs and per-subject layer selection. |
| **Activation analysis** | Multi-layer feature extraction with Sparse Random Projection (k=4096), effective/intrinsic dimensionality, RDM utilities. |
| **Results store + plotting** | All runs deduped into `results.db` (SQLite); per-dataset plotting scripts under `plotters/` produce publication figures from the DB. |

---

## Experiments

Each subdirectory under `experiments/` is a self-contained analysis built on top of the core pipeline.

| Theme | What we ask | Folders |
|---|---|---|
| **Core alignment** | Does coarse supervision beat fine supervision on brain & behavior? | `neurips_2025/`, `representation_analysis/` |
| **Robustness** | Is the effect robust to stimulus choice and splits? | `stimulus_robustness/`, `stimulus_sensitivity/` |
| **Downstream utility** | Do coarse-pretrained features transfer to few-shot, robustness, and continual learning? | `coarse_grain_benefits/`, `continual_learning/` |
| **Interpretability** | What do coarse representations actually encode? | `pca_visualization/`, `model_activating_images/`, `things_visualizations/` |
| **Methodological probes** | How many PCs suffice? What about K > 64? BatchNorm pitfalls? | `reconstruction_analysis/`, `extended_classes/`, `bn_recalibration/` |

---

## Repository structure

```text
visreps/         Main package — run.py (train/eval), trainer, evals, models, dataloaders, analysis
configs/         JSON configs:  train/,  eval/,  grids/
runners/         Local grid runners (train_runner.py, eval_runner.py)
scripts/         PCA label generation, feature extraction, results-DB explorer, smoke tests
plotters/        Per-dataset figure scripts (nsd/, nsd_synthetic/, tvsd/, things/, …)
experiments/     Self-contained analyses (see table above)
pca_labels/      Generated coarse labels (n_classes_{2,4,…,1024}.csv)
results.db       SQLite store: one row per (run, layer, metric)
```

---

## Getting started

<details open>
<summary><b>1. Clone and install</b> (Python 3.11+)</summary>

```bash
git clone git@github.com:yashsmehta/visreps.git
cd visreps
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && source .venv/bin/activate
```

</details>

<details open>
<summary><b>2. Configure paths</b></summary>

```bash
cp .env.example .env   # set IMAGENET_DATA_ROOT, NSD_DATA_DIR, BONNER_DATASETS_HOME
```

</details>

<details open>
<summary><b>3. Train at a chosen granularity</b></summary>

```bash
# Single run: 32 PCA-derived classes
python -m visreps.run --mode train --override pca_labels=true pca_n_classes=32 seed=1

# Sweep granularities and seeds
python runners/train_runner.py --grid configs/grids/train_default.json
```

</details>

<details open>
<summary><b>4. Evaluate alignment</b></summary>

```bash
# RSA on NSD fMRI
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=rsa neural_dataset=nsd

# RSA on THINGS behavioral similarity
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=rsa neural_dataset=things-behavior

# Grid sweep
python runners/eval_runner.py --grid configs/grids/eval_default.json
```

</details>

> Results land in `results.db`; plot with the scripts under `plotters/<dataset>/`.
> Configs in `configs/train/` and `configs/eval/` set defaults; `--override key=value` overrides any field.

---

## Citation

```bibtex
@article{mehta2026coarse,
  title   = {An extremely coarse feedback signal is sufficient for learning human-aligned visual representations},
  author  = {Mehta, Yash and Bonner, Michael F.},
  journal = {arXiv preprint arXiv:2605.05556},
  year    = {2026}
}
```

---

<div align="center">

Licensed under the [MIT License](LICENSE).

</div>
