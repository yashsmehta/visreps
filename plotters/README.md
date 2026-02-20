# Plotters

Publication-quality plotting scripts. All read from `results.db` (SQLite).
Run all scripts from the project root. Outputs organized by dataset under `plotters/figures/`.

## Scripts

### `plot_coarseness.py` — Coarseness progression (main paper figure)
Two figures per (dataset, architecture):
1. **Coarseness bars** — fancy bars with bootstrap 95% CIs (SEM fallback), untrained reference, axis break
2. **Per-subject boxes** — individual subjects connected across class counts (skipped for THINGS)

```bash
python plotters/plot_coarseness.py --dataset nsd   --folder pca_labels_alexnet
python plotters/plot_coarseness.py --dataset tvsd  --folder pca_labels_alexnet
python plotters/plot_coarseness.py --dataset things --folder pca_labels_clip
```

### `plot_architectures.py` — Architecture comparison
Two figures per (dataset, region):
1. **Grouped bars** — all architectures across coarseness levels, 1K baseline line
2. **Per-subject boxes** — architectures at their best coarse cfg, with subject dots

```bash
python plotters/plot_architectures.py --dataset nsd --region "ventral visual stream"
python plotters/plot_architectures.py --dataset tvsd --region IT
```

## Output structure

```
plotters/figures/
├── nsd/          # NSD (human fMRI) figures
├── tvsd/         # TVSD (macaque electrophysiology) figures
└── things/       # THINGS (behavioral similarity) figures
```

## Shared utilities

- `plotter_utils.py` — SQLite helpers (`query_best_scores`, `get_bootstrap_ci`, `get_condition_summary`, `get_subject_scores`), data aggregation, bar plot helpers
