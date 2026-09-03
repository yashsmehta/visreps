# Semantic coarse labels for ImageNet (hand-assigned, no WordNet)

Goal: train CNNs on coarse labels defined by *named* semantic dimensions instead of
PCA median splits, so the unit of labeling is the ImageNet class rather than the image.
Each class is scored 0/1 on several binary dimensions; picking k dimensions gives 2^k labels.

## Key finding: this cluster's ImageNet is not standard ImageNet-1k

`/data/shared/datasets/imagenet` has 1,000 classes, but **354 are not in standard
ILSVRC-2012** (mostly trees, flowers, produce: baobab, bee orchid, blueberry, ...).
Class names must come from `map_clsloc.txt` in that folder, never from torchvision by position.

| block | classes |
|---|---|
| animals (n01, n02) | 327 (only 21 dog breeds, 7 birds) |
| artifacts (n03, n04) | 395 |
| food (n07) | 77 |
| landscapes (n09) | 12 |
| plants, trees, fungi (n11–n13) | 185 |

Consequence: the earlier taxonomic 8-way grouping (dogs / other mammals / birds / ...) does
not transfer (birds = 7 classes) and was dropped.

## Six binary dimensions (definitions in `dimension_definitions.md`)

| dimension | question | fraction = 1 |
|---|---|---|
| natural | grew in nature (1) vs built or cooked (0) | 0.50 |
| handheld | an adult could pick it up (1) vs larger (0) | 0.64 |
| indoor | normally met inside a building (1) vs outside (0) | 0.45 |
| self_moving | moves by its own power: animals, vehicles (1) | 0.29 |
| soft | deforms under hand pressure (1) vs rigid (0) | 0.39 |
| elongated | ≥ ~2.5× longer than wide (1) vs compact (0) | 0.26 |

Strongest correlations: natural/indoor −0.42, indoor/self_moving −0.40, natural/soft +0.38.

## Balance of the eight corners for every triple

Ranked by smallest corner (1000 classes total).

| triple | min | max |
|---|---|---|
| handheld × self_moving × elongated | 35 | 378 |
| handheld × indoor × soft | 27 | 223 |
| **natural × handheld × indoor** | 23 | 257 |
| self_moving × soft × elongated | 18 | 345 |
| all 16 others | < 18 (two have an empty corner) | |

No triple is balanced. The thin corners are real, not labeling noise:
natural+large+indoor = the 23 large dog breeds; manmade+handheld+outdoor = 41 classes of
sports gear, garden tools and weapons. Semantic dimensions are correlated in the world.

## Recommendation

Use **natural × handheld × indoor** (interpretable; first bit matches CLIP PC1) and handle
imbalance in one of two ways:

1. Keep binary labels; use class-weighted loss or balanced sampling during training.
2. Score size and indoorness on a 4-point ordinal scale and split hierarchically at the
   within-branch median (as the PCA pipeline does). Balanced by construction, but bit meaning
   then depends on the branch.

## Files and usage

- `class_dimensions.csv` — one row per class: `class_idx, wnid, class_name` + six 0/1 columns.
- `dimension_definitions.md` — exact questions and anchor examples used for assignment.
- Generate per-image labels (same format as PCA labels) for any dimension set:

```bash
python scripts/coarsegrain/make_semantic_labels.py --dims natural handheld indoor
# -> pca_labels/pca_labels_semantic/n_classes_8.csv
```

Train with `pca_labels=true pca_n_classes=8 pca_labels_folder=pca_labels_semantic`
(grid: `configs/grids/train_semantic.json`). The folder-based loader
(`visreps/dataloaders/obj_cls_folder.py`) reads these CSVs on this cluster.
