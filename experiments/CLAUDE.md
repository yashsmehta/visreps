# Experiments

Self-contained analysis scripts. Each subdirectory has its own analysis, plotting, and data. **Run all scripts from the project root.**

## Task Tracking

Open tasks live in `experiments/todos.md`. Each TODO is numbered (## 1., ## 2., ...). When completing a task, mark it done; when adding, use the next number.

## Key Principle: Reuse Before Rewriting

Two utility modules already exist with shared functions. **Always check these before writing new model-loading, feature-extraction, or label-loading code.**

### `coarse_grain_benefits/utils.py` — General-purpose model + feature utilities

```python
load_model_by_config(cfg_id, seed, checkpoint_dir=None, device=None)
    # Unified loader: checkpoint (32, 64, 1000) or 'pretrained' for torchvision

load_coarse_model(cfg_id, seed, checkpoint_dir=None, device=None)
    # Load trained model by cfg_id + seed, auto-resolves checkpoint path

extract_features(model, loader, layer='fc2', device=None, show_progress=True)
    # -> (features: np.ndarray, labels: np.ndarray). Handles conv flattening.

extract_features_batch(model, images, layer='fc2', device=None)
    # Single-batch extraction -> np.ndarray

get_feature_extractor(model, layers)
    # Wrap model with FeatureExtractor (post-ReLU). layers: str or list.

get_model_configs(cfg_ids=None, seeds=None, include_pretrained=False)
    # -> list of (cfg_id, seed) tuples for experiment sweeps

get_config_name(cfg_id, seed)
    # -> human-readable string, e.g. "32-waya" or "Torchvision Pretrained"
```

### `representation_analysis/utils.py` — Multi-layer extraction + label loading

```python
extract_layer(model, loader, device, layer=None)
    # -> np.ndarray of features from one layer (default: fc2)

extract_all_layers(model, loader, device, layers=None, conv_pool_size=3)
    # -> dict[layer_name, np.ndarray]. Pools conv layers to conv_pool_size x conv_pool_size.

load_labels(loader)
    # -> (pca_labels, sem_labels, synsets, img_paths). Matches to dataset order.

load_data_and_models(device=None)
    # Composite: loads dataset + labels + pretrained & 32-way models, extracts fc2.

load_data_and_models_all_layers(device=None, layers=None)
    # Same but extracts all layers for both models.
```

### `wordnet/` — Semantic labels and WordNet hierarchy

```python
# wordnet_utils.py
setup()  # Ensure NLTK WordNet is downloaded

# make_semantic_labels.py
SUPER_CATEGORIES  # dict: 8 super-categories -> list of Level-6 synset names
```

## Directory Overview

| Directory | Purpose | Key outputs |
|-----------|---------|-------------|
| `neurips_2025/` | NeurIPS submission figures (fig1-fig4) | Publication PNGs |
| `1k_pretrained/` | RSA of pretrained models (AlexNet, ViT) on NSD | `logs/1k_pretrained_nsd_rsa.csv` |
| `coarse_grain_benefits/` | Downstream benefits: few-shot, robustness, curriculum, linear probe | `results/*.csv` |
| `pca_visualization/` | PCA structure: PC pole images, semantic enrichment analysis | `figures/`, `pc_histogram/` |
| `representation_analysis/` | Dimensionality, nearest neighbors, variance, task-brain alignment | `figs/`, `dimensionality/` |
| `reconstruction_analysis/` | PC reconstruction quality vs brain alignment. Shared plotting logic in `plot_utils.py`; `plot.py` (coarse baseline) and `plot_case_study.py` (2-class baseline) are thin wrappers. | `figures/` |
| `wordnet/` | WordNet hierarchy utilities and semantic label generation | `semantic_categories.csv` |
| `model_activating_images/` | Most-activating ImageNet images per output class (logit ranking) | `rankings.csv`, `figures/` |
| `stimulus_robustness/` | Coarse > fine alignment robust to stimulus subsampling | `data.json`, PNGs |
| `stimulus_sensitivity/` | k-fold CV RSA fluctuation analysis | `data.json`, PNGs |
