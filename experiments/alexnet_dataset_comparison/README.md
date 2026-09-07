# AlexNet: ILSVRC-2010 versus ILSVRC-2012 (ImageNet-1k)

Dataset provenance supplied by Yash Mehta: the historical `imagenet/`
directory is the complete ILSVRC-2010 training release (1,261,406 images,
1000 synsets, verified against the 2010 devkit). It is not the ILSVRC-2012
ImageNet-1k dataset used by torchvision checkpoints. The releases share
640 of 1000 classes; 2010 has no per-class image cap. Previous CNN models
trained from this directory should be described as **ILSVRC-2010-trained**.

On lab-1 the canonical paths are `/data/shared/datasets/ilsvrc2010/` and
`/data/shared/datasets/ilsvrc2012/`. On Rockfish they are under
`/scratch4/mbonner5/shared/`. Old paths remain symlinks on those machines.
This experiment was started on lab-2 (`cogsci-mb-gpu2`), which still uses
`/data/shared/datasets/imagenet/` and does not have the new 2012 directory.
The 2010 config keeps that working lab-2 path; override `dataset_path`
with the canonical directory when running elsewhere.

Train torchvision AlexNet from scratch using the hyperparameters saved in
`/data/ymehta3/default/cfg1000a/config.json`: 20 epochs, batch size 32,
AdamW (LR 0.0005, weight decay 0.001), gradient clipping 1.0, two warmup
epochs, cosine decay, AMP, dropout 0.3, and mild augmentation
(resize 256, center crop 224, horizontal flip, rotation ±10 degrees).
Label smoothing is explicitly 0.1, the trainer's default for that configuration.
The optimizer excludes biases from weight decay as in the previous training code.

The model is torchvision's AlexNet with 6×6 pooling, its standard channel
widths and initialization, 1000 output classes, and no BatchNorm. Its
configurable dropout probability is set to the previous CNN's 0.3 rather
than torchvision's default 0.5. Both runs use seed 1.

ImageNet-2010 retains the previous folder backend's fixed 80/20 image split
(split seed 42: 1,009,124 train / 252,282 held-out images) and existing label map.
The official 50,000-image 2010 validation set is separate (`ilsvrc2010_val/`
on lab-1); this historical replication does not use it. Likewise,
`ilsvrc2010_trainsplit/val` is held-out training data, not official validation.
ImageNet-2012 uses its official
train/validation splits with labels derived from its own synset folders;
do not reuse the 2010 label map. Thus training-set sizes, optimizer-step
counts, and classification validation sets differ; compare THINGS using
the same held-out concepts. This is a dataset-and-split comparison.

From the repository root:

```bash
.venv/bin/python -m visreps.run --mode train --config experiments/alexnet_dataset_comparison/common.json experiments/alexnet_dataset_comparison/imagenet2010.json

.venv/bin/python -m visreps.run --mode train --config experiments/alexnet_dataset_comparison/common.json experiments/alexnet_dataset_comparison/imagenet2012.json
```

The 2012 path must contain `train/<wnid>/*` and `val/<wnid>/*` with the same
1000 synsets (1,281,167 train and 50,000 validation images). The default path
is for lab-1; on Rockfish add
`--override dataset_path=/scratch4/mbonner5/shared/ilsvrc2012`.
Use `val/`, not the duplicate hard-linked `val_flat/` directory.
Never point a 2010 manifest at 2012 data: this loader derives the 2012 class
mapping independently and requires matching train/val synsets.
The explicit folder backend prevents a
machine's optional parquet package from changing the chosen dataset.

Checkpoints are saved under `model_checkpoints/alexnet_imagenet2010_cnn_recipe/cfg1000a`
and `model_checkpoints/alexnet_imagenet2012_cnn_recipe/cfg1000a` at epochs
0, 10, and 20. Use a new checkpoint directory or seed for subsequent runs;
the general trainer does not protect existing checkpoints from overwrites.
