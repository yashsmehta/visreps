# Dataset-specific BatchNorm proof of principle

Dataset recalibration does not change the model ranking in any of the five regions. CLIP-32 leads in TVSD V1 and NSD ventral; default-1000 leads in TVSD V4/IT and NSD early. Recalibration modestly raises most mean scores. The exception is CLIP-32 in NSD early (0.2108 to 0.1716), where the selected layer changes from conv4 to fc1. Default-1000 NSD ventral also changes selected layer, from fc1 to fc2. The CLIP-32 NSD ventral advantage persists with saved ImageNet statistics (+0.0489) and recalibrated statistics (+0.0534).

CustomCNN, seed 1, epoch 20. CLIP-derived 32-class checkpoint: `/data/ymehta3/clip_pca/cfg32a/checkpoint_epoch_20.pth`; default 1000-class checkpoint: `/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth`.

Scores are Spearman RSA, averaged over subjects (TVSD: 2; NSD: 8). Saved ImageNet means the original checkpoint running statistics, without any additional calibration. Dataset recalibration uses the current evaluator and its matching cached statistics, computed from dataset training images only, excluding every participating subject’s test set. Both conditions use evaluation mode with dropout disabled and identical model weights.

| Dataset / region | Model | Saved ImageNet BN | Dataset BN | Change |
|---|---|---:|---:|---:|
| NSD / early visual stream | clip32 | 0.2108 | 0.1716 | -0.0392 |
| NSD / early visual stream | default1000 | 0.2285 | 0.2383 | +0.0098 |
| NSD / ventral visual stream | clip32 | 0.2971 | 0.3156 | +0.0186 |
| NSD / ventral visual stream | default1000 | 0.2482 | 0.2623 | +0.0141 |
| TVSD / IT | clip32 | 0.1488 | 0.1505 | +0.0017 |
| TVSD / IT | default1000 | 0.1707 | 0.1773 | +0.0066 |
| TVSD / V1 | clip32 | 0.1607 | 0.1721 | +0.0114 |
| TVSD / V1 | default1000 | 0.1031 | 0.1097 | +0.0066 |
| TVSD / V4 | clip32 | 0.2056 | 0.2101 | +0.0045 |
| TVSD / V4 | default1000 | 0.2381 | 0.2414 | +0.0033 |

Model advantage (CLIP-32 minus default-1000):

| Dataset / region | Saved ImageNet BN | Dataset BN |
|---|---:|---:|
| NSD / early visual stream | -0.0177 | -0.0667 |
| NSD / ventral visual stream | +0.0489 | +0.0534 |
| TVSD / IT | -0.0219 | -0.0268 |
| TVSD / V1 | +0.0575 | +0.0623 |
| TVSD / V4 | -0.0324 | -0.0313 |

Selected layers:

| Dataset / region | Model | BN condition | Layer |
|---|---|---|---|
| TVSD / V1 | clip32 | saved_imagenet | conv3 |
| TVSD / V4 | clip32 | saved_imagenet | fc1 |
| TVSD / IT | clip32 | saved_imagenet | fc1 |
| TVSD / V1 | clip32 | dataset_recalibrated | conv3 |
| TVSD / V4 | clip32 | dataset_recalibrated | fc1 |
| TVSD / IT | clip32 | dataset_recalibrated | fc1 |
| TVSD / V1 | default1000 | saved_imagenet | fc1 |
| TVSD / V4 | default1000 | saved_imagenet | fc1 |
| TVSD / IT | default1000 | saved_imagenet | fc1 |
| TVSD / V1 | default1000 | dataset_recalibrated | fc1 |
| TVSD / V4 | default1000 | dataset_recalibrated | fc1 |
| TVSD / IT | default1000 | dataset_recalibrated | fc1 |
| NSD / early visual stream | clip32 | saved_imagenet | conv4 |
| NSD / ventral visual stream | clip32 | saved_imagenet | fc1 |
| NSD / early visual stream | clip32 | dataset_recalibrated | fc1 |
| NSD / ventral visual stream | clip32 | dataset_recalibrated | fc1 |
| NSD / early visual stream | default1000 | saved_imagenet | conv4 |
| NSD / ventral visual stream | default1000 | saved_imagenet | fc1 |
| NSD / early visual stream | default1000 | dataset_recalibrated | conv4 |
| NSD / ventral visual stream | default1000 | dataset_recalibrated | fc2 |

Protocol: identical preprocessing, subject sets, selection/test splits, and extraction settings. Layer selection is repeated independently for each BN condition, with one shared layer per region selected using mean selection RSA across subjects. Thus these are end-to-end evaluation effects, which can include a selected-layer change. Selection uses up to 1000 training stimuli per subject and fixed sparse random projection; test scores use exact selected-layer features. No PCA reconstruction, bootstrap, or significance tests. TVSD has 100 shared test images; NSD has 907.

Checks: learned-parameter hashes were unchanged by BN handling; saved-statistics buffers were preserved; recalibrated buffers differed; all modules were in evaluation mode. Each condition loads its checkpoint afresh. JSON metadata records hashes, full effective configuration, calibration-image counts, and calibration identities. Results contain all expected subject/region pairs.

No database writes: `log_expdata=false`, the database writer was disabled within the experiment process, and `results.db` SHA-256 was identical before and after all evaluations. Original checkpoints were not written.

Run: `OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python -u experiments/bn_recalibration/compare_dataset_stats.py`. Existing completed JSON results are skipped. Summarize: `.venv/bin/python experiments/bn_recalibration/summarize_dataset_stats.py`.

This tests retaining training-time ImageNet statistics versus target-dataset recalibration. It does not test recomputing statistics on a fresh ImageNet calibration set. The single-seed results do not establish generality across training seeds.
