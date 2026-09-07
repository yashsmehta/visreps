"""Summarize the local BN proof-of-principle results."""
import json
from pathlib import Path
import pandas as pd

p = Path('experiments/bn_recalibration/dataset_stats_comparison')
s = pd.read_csv(p / 'summary.csv')
w = s.pivot(index=['dataset', 'region', 'model'], columns='bn_condition', values='mean_score')
w['delta_recal_minus_saved'] = w.dataset_recalibrated - w.saved_imagenet
w.to_csv(p / 'bn_effects.csv')
gaps = s.pivot(index=['dataset', 'region', 'bn_condition'], columns='model', values='mean_score')
gaps['clip32_minus_default1000'] = gaps.clip32 - gaps.default1000
gaps.to_csv(p / 'model_gaps.csv')
lines = ['# Dataset-specific BatchNorm proof of principle', '',
'CustomCNN, seed 1, epoch 20. CLIP-derived 32-class checkpoint: `/data/ymehta3/clip_pca/cfg32a/checkpoint_epoch_20.pth`; default 1000-class checkpoint: `/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth`.', '',
'Scores are Spearman RSA, averaged over subjects (TVSD: 2; NSD: 8). Saved ImageNet means the original checkpoint running statistics, without any additional calibration. Dataset recalibration uses the current evaluator and its matching cached statistics, computed from dataset training images only, excluding every participating subject’s test set. Both conditions use evaluation mode with dropout disabled and identical model weights.', '',
'| Dataset / region | Model | Saved ImageNet BN | Dataset BN | Change |',
'|---|---|---:|---:|---:|']
for (dataset, region, model), row in w.iterrows():
    lines.append(f'| {dataset.upper()} / {region} | {model} | {row.saved_imagenet:.4f} | {row.dataset_recalibrated:.4f} | {row.delta_recal_minus_saved:+.4f} |')
lines += ['', 'Model advantage (CLIP-32 minus default-1000):', '',
'| Dataset / region | Saved ImageNet BN | Dataset BN |', '|---|---:|---:|']
for (dataset, region), g in gaps.groupby(level=['dataset', 'region'], sort=False):
    v = g.droplevel(['dataset', 'region']).clip32_minus_default1000
    lines.append(f'| {dataset.upper()} / {region} | {v.saved_imagenet:+.4f} | {v.dataset_recalibrated:+.4f} |')
lines += ['', 'Selected layers:', '', '| Dataset / region | Model | BN condition | Layer |', '|---|---|---|---|']
for row in s.itertuples():
    lines.append(f'| {row.dataset.upper()} / {row.region} | {row.model} | {row.bn_condition} | {row.layer} |')
lines += ['',
'Protocol: identical preprocessing, subject sets, selection/test splits, and extraction settings. Layer selection is repeated independently for each BN condition, with one shared layer per region selected using mean selection RSA across subjects. Thus these are end-to-end evaluation effects, which can include a selected-layer change. Selection uses up to 1000 training stimuli per subject and fixed sparse random projection; test scores use exact selected-layer features. No PCA reconstruction, bootstrap, or significance tests. TVSD has 100 shared test images; NSD has 907.', '',
'Checks: learned-parameter hashes were unchanged by BN handling; saved-statistics buffers were preserved; recalibrated buffers differed; all modules were in evaluation mode. Each condition loads its checkpoint afresh. JSON metadata records hashes, full effective configuration, calibration-image counts, and calibration identities. Results contain all expected subject/region pairs.', '',
'No database writes: `log_expdata=false`, the database writer was disabled within the experiment process, and `results.db` SHA-256 was identical before and after all evaluations. Original checkpoints were not written.', '',
'Run: `OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python -u experiments/bn_recalibration/compare_dataset_stats.py`. Existing completed JSON results are skipped. Summarize: `.venv/bin/python experiments/bn_recalibration/summarize_dataset_stats.py`.', '',
'This tests retaining training-time ImageNet statistics versus target-dataset recalibration. It does not test recomputing statistics on a fresh ImageNet calibration set. The single-seed results do not establish generality across training seeds.']
(p / 'comparison.md').write_text('\n'.join(lines) + '\n')
print(w.to_string())
print(gaps[['clip32_minus_default1000']].to_string())
