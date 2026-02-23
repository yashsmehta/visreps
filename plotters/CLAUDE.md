# Plotters

Publication-quality plotting scripts reading from `results.db` (SQLite). Run from project root. Figures save to `plotters/<dataset>/figures/`.

## Scripts

**Coarseness progression** — per-dataset scripts with shared logic in `plot_helpers.py`:
```bash
python plotters/nsd/plot_coarseness.py --folder pca_labels_alexnet
python plotters/nsd/plot_coarseness_finegrained.py --folder pca_labels_clip
python plotters/nsd_synthetic/plot_coarseness.py --folder pca_labels_alexnet
python plotters/tvsd/plot_coarseness.py --folder pca_labels_alexnet
python plotters/things/plot_coarseness.py --folder pca_labels_alexnet
```
*Note: All coarseness plotters default to RSA with Spearman correlation. To plot Encoding Score, add `--analysis encoding_score --compare_method pearson`.*

**Architecture comparison** — `plot_architectures.py`:
```bash
python plotters/plot_architectures.py --dataset nsd --region "ventral visual stream"
```

**Shared utilities:** `plotter_utils.py` (DB queries, aggregation), `plot_helpers.py` (bar/box drawing).

---

## `results.db` Schema

4 tables, all keyed on `run_id` (SHA256[:12] of experiment identity fields, including `compare_method`). **Never modify directly.**

### `results` — 3,856 rows, one per evaluated run (best layer only)

```
run_id              TEXT NOT NULL   ─┐
compare_method      TEXT NOT NULL    │ UNIQUE together with layer
layer               TEXT NOT NULL   ─┘  best layer name, e.g. "conv5_pre"
score               REAL              final test score
ci_low, ci_high     REAL              bootstrap 95% CI bounds
analysis            TEXT NOT NULL     "rsa" | "encoding_score"
seed                INTEGER NOT NULL  1, 2, or 3
epoch               INTEGER NOT NULL  0 (untrained) | 20 (trained)
region              TEXT              see coverage table; "N/A" for THINGS
subject_idx         TEXT              "0"–"7" (NSD), "0"–"1" (TVSD), "N/A" (THINGS)
neural_dataset      TEXT NOT NULL     "nsd" | "nsd_synthetic" | "tvsd" | "things-behavior"
cfg_id              INTEGER           training classes: 2, 4, 8, 16, 32, 64, or 1000
pca_labels          BOOLEAN NOT NULL  0 = 1K baseline, 1 = PCA coarse labels
pca_n_classes       INTEGER           2–64 (always 2 for baseline rows)
pca_labels_folder   TEXT              "imagenet1k" | "pca_labels_{alexnet,clip,dino,vit}"
model_name          TEXT NOT NULL     always "CustomCNN"
checkpoint_dir      TEXT              /data/ymehta3/{default,alexnet_pca,clip_pca,dino_pca,vit_pca}
reconstruct_from_pcs BOOLEAN          0 = standard, 1 = PC reconstruction (pca_k 1–15, cfg_id=1000 only)
pca_k               INTEGER           PCs for reconstruction; default 1 for standard runs
```

**Conventions:** Baseline rows use `pca_labels=0, pca_labels_folder="imagenet1k", cfg_id=1000`. RSA uses `compare_method="spearman"`; encoding uses `"pearson"`.

### `run_configs` — 3,856 rows, 1:1 with results

`run_id` (PK) → `config_json` (full 44-key JSON: training params, arch, eval settings) + `created_at`.

### `bootstrap_distributions` — 3,856 rows, 1:1 with results

`(run_id, compare_method)` UNIQUE → `scores` (JSON array, 1,000 bootstrap iterations, 90% subsample each).

### `layer_selection_scores` — 30,086 rows (2,149 runs × 14 layers)

`(run_id, compare_method, layer)` UNIQUE → `score`. Layers: `conv{1-5}_{pre,post}`, `fc{1-2}_{pre,post}` (`_pre` = before activation/batchnorm, `_post` = after).

### Data Coverage

| Dataset | Regions | Subjects | Analyses | Rows (std) |
|---------|---------|----------|----------|------------|
| `nsd` | `early visual stream`, `ventral visual stream`, `V1`, `V2`, `V3`, `hV4`, `FFA`, `PPA` | 8 (0–7) | rsa, encoding | ~1,369 |
| `nsd_synthetic` | `V1`, `V2`, `V3`, `hV4`, `FFA`, `PPA` | 8 (0–7) | rsa | 672 |
| `tvsd` | `V1`, `V4`, `IT` | 2 (0–1) | rsa, encoding | 702 |
| `things-behavior` | `N/A` | `N/A` | rsa | 78 |

PCA architectures: AlexNet and CLIP have broad coverage; DINO and ViT are NSD-only with fewer runs. PC reconstruction (`reconstruct_from_pcs=1`): 1,035 rows sweeping `pca_k` 1–15 on the 1K baseline across NSD/TVSD/THINGS.

### Common Queries

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("results.db")

# RSA scores for one region (exclude PC reconstruction)
pd.read_sql("SELECT * FROM results WHERE region='ventral visual stream' AND compare_method='spearman' AND reconstruct_from_pcs=0", conn)

# Per-subject scores for a specific condition
pd.read_sql("SELECT subject_idx, seed, score FROM results WHERE neural_dataset='nsd' AND region='ventral visual stream' AND pca_labels_folder='pca_labels_alexnet' AND cfg_id=32 AND compare_method='spearman' AND epoch=20 AND reconstruct_from_pcs=0", conn)

# Full config for a run
import json; json.loads(conn.execute("SELECT config_json FROM run_configs WHERE run_id=?", ("223c4cfff8a7",)).fetchone()[0])
```
