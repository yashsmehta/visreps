# Fine-class manifold SNR experiment

`run_coarsegrain_manifolds.py` implements the ImageNet-1k experiment described
in the August 22 plan.

- Models: CLIP-label checkpoints at K = 2, 4, 8, 16, 32, and 64, plus the
  conventionally trained 1,000-way checkpoint; seeds a, b, and c.
- Representation: the CustomCNN second hidden fully-connected layer (AlexNet
  FC7), after batch normalization and ReLU and before the classifier. The
  4,096-dimensional representation is used without projection.
- Evaluation: 50 images per original ImageNet-1k class, selected once from the
  legacy loader's deterministic 20% training-image holdout. Training labels
  never define evaluation manifolds.
- Sampling: 100 reproducible panels of 50 classes, shared by every model.
- Metric: all 2,450 directed pairs at m = 5; m = 1 and 10 are sensitivity
  analyses. Negative SNRs and pair direction are preserved.
- Validation: 25-vs-50-image stability and empirical five-shot
  nearest-prototype error on a fixed subset of pairs.

Coarse labels are used only to identify how each checkpoint was trained. They
are never loaded by the analysis, never used to regroup the evaluation images,
and never used to filter class pairs. In particular, this is not a
within-coarse-cluster SNR analysis.

The JSON output contains panel and aggregate summaries. Per-pair SNR, predicted
error, Signal, Bias, dimension noise, both signal-noise terms, and noise-noise
overlap are stored in one NPZ per model. Retention uses the repetition-matched
1,000-way model with the same seed.

## Run

```bash
python experiments/manifold_analysis/run_coarsegrain_manifolds.py \
  --output-dir experiments/manifold_analysis/snr_results
```

Runs resume at model granularity. Use a fresh output directory when changing
sampling parameters. Activation caches are large because they contain FC7 for
all 50,000 validation images.

The default source is
`experiments/manifold_analysis/heldout_dataset/manifest.json`. It contains
50,000 existing image paths (1,000 classes x 50 images) and does not duplicate
the image files. Rebuild it deterministically with:

```bash
python experiments/manifold_analysis/build_heldout_dataset.py
```
