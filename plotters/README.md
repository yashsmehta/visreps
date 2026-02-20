# Plotters

Publication-quality plotting scripts. Most read from `results.db` or pre-computed
CSVs/JSON. Run all scripts from the project root. Outputs go to `plotters/figures/`.

## Current paper figures
- `coarseness_barplot.py` — RSA across all label granularity levels (bar + axis-break)

## Architecture comparisons
- `bar_plot_comparison.py` — All architectures bar plots (MODE: `rsa` or `semantic`)
- `box_plot_architectures.py` — Architecture box plots (METRIC: `rsa` or `encoding`)
- `paired_comparison.py` — Paired t-test plots (MODE: `coarse_vs_fine` or `architectures`)

## Per-metric figures
- `rsa_score_by_classes.py` — RSA scores vs class granularity (box + lines)
- `encoding_score_by_classes.py` — Encoding scores vs class granularity (bar plots)
- `tvsd_coarse_vs_fine_barplots.py` — Macaque TVSD analysis

## Shared utilities
- `plotter_utils.py` — Data aggregation, best-layer selection, bar plot helpers
