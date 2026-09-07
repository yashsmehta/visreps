# AlexNet RSA comparison — 2026-09-06

The ImageNet-2010 model completed epoch 20 (seed 1), with 26.7867% top-1 and 48.5671% top-5 accuracy on its held-out training-release split. Checkpoint: `model_checkpoints/alexnet_imagenet2010_cnn_recipe/cfg1000a/checkpoint_epoch_20.pth`. Its saved config records `imagenet_version=2010`.

Comparison: torchvision AlexNet `IMAGENET1K_V1` (`alexnet-owt-7be5be79.pth`). Both models were evaluated afresh with the current repository evaluation pipeline.

| Evaluation | ImageNet-2010, epoch 20 | Torchvision ImageNet-2012 | Delta (2010 − 2012) | Relative change | Selected layers (2010 / 2012) |
|---|---:|---:|---:|---:|---|
| NSD: early visual stream | 0.1650 | 0.2200 | -0.0549 | -25.0% | fc1 / conv4 |
| NSD: ventral visual stream | 0.1934 | 0.2842 | -0.0908 | -31.9% | fc1 / fc2 |
| THINGS behavior | 0.4078 | 0.4916 | -0.0838 | -17.0% | fc1 / fc2 |

The 2010 checkpoint scores lower in both NSD regions for all eight subjects.

Method: Spearman correlation between Pearson-distance RDMs; post-activation features; identical preprocessing and selection/evaluation splits. Selection uses the fixed seed-42 sparse random projection (up to 4096 dimensions); held-out scores use exact features without projection or PCA reconstruction. NSD uses the default NCSNR-filtered data, 1000 selection images per subject, a shared selected layer per region, and 907 shared test stimuli. Reported NSD scores are arithmetic means over eight subjects. THINGS uses concept-averaged image features and behavioral embeddings, with 370 selection and 1484 evaluation concepts (split seed 42).

Interpretation: These are descriptive score differences from one trained seed; no bootstrap or significance test was run. This comparison changes dataset, split, and training recipe together, so it cannot isolate an ImageNet-version effect. A recipe-matched ImageNet-2012 checkpoint is not present locally.

Reproduce from the repository root: `OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python experiments/alexnet_dataset_comparison/evaluate_rsa.py`. The runner skips completed result JSON files; move those files before a fresh rerun. Raw scores, selection scores, CLI overrides, and the evaluation log are saved alongside this report.
