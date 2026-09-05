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
| indoor | normally met inside a building (1) vs outside (0), judged by the typical ImageNet photo | 0.41 |
| self_moving | moves by its own power: animals, vehicles (1) | 0.29 |
| soft | deforms under hand pressure (1) vs rigid (0) | 0.39 |
| elongated | ≥ ~2.5× longer than wide (1) vs compact (0) | 0.26 |

Strongest correlations: natural/indoor −0.43, indoor/self_moving −0.34, natural/soft +0.38.

## Balance of the eight corners for every triple

Ranked by smallest corner (1000 classes total).

| triple | min | max |
|---|---|---|
| handheld × self_moving × elongated | 35 | 376 |
| natural × indoor × soft | 31 | 240 |
| handheld × indoor × soft | 27 | 215 |
| **natural × handheld × indoor** | 23 | 275 |
| self_moving × soft × elongated | 18 | 345 |
| all 15 others | ≤ 17 (one has an empty corner) | |

No triple is balanced. The thin corners are real, not labeling noise:
natural+large+indoor = the 23 medium/large dog breeds (the only large natural things that
live in homes); manmade+handheld+outdoor = sports gear, garden tools, cameras and outdoor
garments. Semantic dimensions are correlated in the world.

## Chosen labels: natural × handheld × indoor (`n_classes_8.csv`)

Label = 4·natural + 2·handheld + indoor. Regenerated 2026-09-05 over all 1,261,406 images
(same image list as the PCA label files; the 1,000 wnids match the on-disk ILSVRC-2010 folders).

| label | bits | meaning | classes | images |
|---|---|---|---|---|
| 0 | 000 | manmade · large · outdoor (vehicles, buildings) | 127 | 170,136 |
| 1 | 001 | manmade · large · indoor (furniture, appliances, shop interiors) | 82 | 107,785 |
| 2 | 010 | manmade · handheld · outdoor (sports gear, garden tools, cameras, outdoor garments) | 64 | 77,997 |
| 3 | 011 | manmade · handheld · indoor (household objects, tools) | 231 | 297,581 |
| 4 | 100 | natural · large · outdoor (large wild animals, trees, landscapes) | 124 | 152,107 |
| 5 | 101 | natural · large · indoor (medium/large dog breeds) | 23 | 38,650 |
| 6 | 110 | natural · handheld · outdoor (insects, small wild animals, wildflowers, berries on the plant) | 275 | 326,242 |
| 7 | 111 | natural · handheld · indoor (produce as bought, small pets, houseplants) | 74 | 90,908 |

Audited 2026-09-04 against sample images for ambiguous names (balloon = hot-air balloon,
bell = church bell, crane = construction crane, brass = memorial plaque).

**Audit of natural, handheld, indoor (2026-09-05).** Every class was re-checked by hand against
the montages in `figures/dim_{natural,handheld,indoor}.jpg` (one photo per class, left = 0,
right = 1; `scripts/coarsegrain/plot_semantic_dimensions.py`) and against CLIP ViT-L/14
zero-shot indoor/outdoor scores on 64 photos per class (`scripts/coarsegrain/audit_indoor_clip.py`
→ `audit_indoor_clip.csv`). CLIP is a flag, not an oracle: it calls aquarium animals, museum
armour and skyscrapers "indoor". Indoor changed for 66 classes: 21 outdoor garments, 8 pieces
of outdoor gear (cameras, binoculars, padlock, seat belt, shopping cart) and 28 produce
classes that this dataset photographs on the plant went to outdoor; bakery, cinema, garage,
greenhouse, prison, subway train, cash machine, cockroach, moth orchid, axolotl, volleyball,
and the food classes mislabeled as plants (peanut n12, olive n12) went to indoor. Dog breeds
stay indoor: they are the only large natural things that live in homes, and CLIP puts them
at 0.2–0.6 indoor versus < 0.1 for every wild animal. Handheld changed for 4 classes
(olive, gibbon, raccoon, barracouta → 1). Natural needed no change.

**Checkpoints in `/data/ymehta3/semantic/cfg8{a,b,c}` were trained on the pre-audit
`n_classes_8.csv`** (git history before 2026-09-05); retrain to use the corrected labels.

Imbalance is left as is (plain cross-entropy), matching the PCA-label runs. The alternative
handheld × indoor × soft is better balanced but its corners mix trees, trucks and fences, and
it drops the natural/manmade split that matches CLIP PC1.

## Training

`configs/grids/train_semantic.json` pins every hyperparameter to the values saved in the
PCA coarse checkpoints (`/data/ymehta3/alexnet_pca/cfg*/config.json`): batch size 32,
32 workers, AdamW lr 5e-4, wd 1e-3, 20 epochs, 2 warmup, cosine, grad clip 1.0, AMP,
augmentation on, CustomCNN with dropout 0.3 and batchnorm. Checkpoints go to
`/data/ymehta3/semantic/cfg8{a,b,c}`.

```bash
python runners/train_runner.py --grid configs/grids/train_semantic.json \
    --arch configs/train/architectures/custom_cnn.json
```

## Files and usage

- `class_dimensions.csv` — one row per class: `class_idx, wnid, class_name` + six 0/1 columns.
- `dimension_definitions.md` — exact questions and anchor examples used for assignment.
- `n_classes_8.csv` — per-image labels for natural × handheld × indoor (see above).
- Regenerate, or build labels for any other dimension set:

```bash
python scripts/coarsegrain/make_semantic_labels.py --dims natural handheld indoor
# -> pca_labels/pca_labels_semantic/n_classes_8.csv
```

Train with `pca_labels=true pca_n_classes=8 pca_labels_folder=pca_labels_semantic`
(grid: `configs/grids/train_semantic.json`). The folder-based loader
(`visreps/dataloaders/obj_cls_folder.py`) reads these CSVs on this cluster.
