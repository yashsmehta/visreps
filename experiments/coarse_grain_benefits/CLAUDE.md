# Coarse-Grain Benefits Experiments

Evaluates downstream benefits of coarse-grained pre-training beyond brain alignment.

## Experiments

| Script | Purpose | Metric |
|--------|---------|--------|
| `few_shot_learning.py` | Transfer to CIFAR-100 with k-shot linear probes | Classification accuracy |
| `imagenet_c_robustness.py` | Robustness to ImageNet-C corruptions | Relative accuracy drop |
| `augmentation_invariance.py` | Representation stability under OOD augmentations | Cosine similarity |
| `curriculum_finetuning.py` | Coarse→1000-way curriculum training | Top-1/Top-5 accuracy |

## Usage

```bash
# All scripts support these common args:
--cfg_ids 32 64 1000      # Granularities to compare
--seeds 1                  # Training seeds (1-3)
--layer fc2               # Layer for feature extraction
--include_pretrained      # Also evaluate torchvision AlexNet

# Few-shot learning
python few_shot_learning.py --k_shots 1 5 10 20 --n_trials 5

# Corruption robustness (requires: pip install imagecorruptions)
python imagenet_c_robustness.py --severity 3 --n_images 5000
python imagenet_c_robustness.py --all_corruptions  # Full 15 corruptions

# Augmentation invariance (uses albumentations for OOD transforms)
python augmentation_invariance.py --n_images 1000 --n_augments 10

# Curriculum fine-tuning
python curriculum_finetuning.py --coarse_cfg_id 64 --transfer_mode full
# Transfer modes: full | late_layers | fc_only | head_only
```

## Shared Utilities (`utils.py`)

- `load_model_by_config(cfg_id, seed)` — Load checkpoint or torchvision pretrained
- `extract_features(model, loader, layer)` — Frozen feature extraction
- `get_model_configs(cfg_ids, seeds)` — Generate (cfg_id, seed) pairs

## Output

Results saved to `results/` as CSV files with columns for cfg_id, seed, layer, and experiment-specific metrics.

## Dependencies

- `imagecorruptions` — for ImageNet-C experiment
- `albumentations` — for augmentation invariance
- `sklearn` — for linear probes
