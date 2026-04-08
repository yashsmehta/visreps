# Manuscript Figures

6 main figures. Each lives in its own `fig{N}/` subfolder. Shared utilities in `fig_utils.py` and `things_utils.py`. All figures save at 300 DPI.

Each figure directory contains a `figure{N}_description.md` file with the Nature-style figure caption — a self-contained description of the figure that a reader could understand by looking at the figure alongside the caption.

## Figure 1: Schematic (`fig1/`)

Method and experimental pipeline overview. The schematic itself is assembled externally (vector editor).

- **1a** (`figure1a.png`): Label space PCA scatter — 2/4/1000-way coloring of shared CLIP PCA coordinates with image insets.

**Files:**
- `plot_label_space.py` — generates figure1a
- `utils.py` — constants (palettes, inset classes), style, median split labels, image inset overlay
- `compute_pca_cache.py` — extracts CLIP features and caches PCA projections (run with `--recompute`)

**Data:** `pc_scatter_1per_class.npz` (cached PCA coordinates, 1 image per class), ImageNet images via `IMAGENET_DATA_DIR`.

## Figure 2: Categorical Nature of Representations (`fig2/`)

Shows how coarse vs fine-grained labels shape learned representations.

- **2b** (`figure2.png`): Learned representation mosaic — FC1 PCA projections of CNN trained on 1000 classes vs 4 derived classes, with image thumbnails at per-class centroid positions.

**Files:**
- `figure2.py` — orchestrator
- `plot_representations.py` — generates figure2
- `utils.py` — image thumbnails, PCA sign-flip alignment, style

**Data:** `experiments/representation_analysis/2pcs_compare/data_4way_alexnet.npz`, ImageNet images via `IMAGENET_DATA_DIR`.

## Figure 3: Neural Alignment Across Species (`fig3/`)

TVSD (macaque electrophysiology) + NSD (human fMRI) alignment across coarseness levels.

**Layout:** 2 rows x 4 columns + schematic row. Columns grouped by dataset (TVSD | NSD), within each pair: alignment-per-bit | raw Spearman rho. Rows: early visual cortex | higher visual cortex.

**Files:**
- `figure3.py` — main assembly
- `panel_raw.py` — raw Spearman rho scatter (all label sources)
- `panel_efficiency.py` — CLIP-only alignment-per-bit
- `schematic_utils.py` — dataset schematics + brain insets
- `shared.py` — style constants, data fetching, axis formatting
- `generate_brain_renders.py` — pre-renders brain region images
- `extract_class_centroids.py` — class centroid data for schematics

**Data:** `results.db` (NSD/TVSD results), `class_centroids_alexnet.npz`, `assets/` (brain renders, species icons).

## Figure 4: THINGS Behavioral Alignment (`fig4/`)

Coarse-grained training vs human behavioral similarity (THINGS dataset).

**Layout:** Row 1: Schematic | Coarseness | Model Comparison | Data Efficiency. Row 2: 4 PCA scatter panels.

**Panels:**
- **A** (schematic): THINGS behavioral similarity task placeholder
- **B** (coarseness): RSA vs training classes (2-64), 3 label sources (AlexNet, CLIP, Pixels), 1000-way dashed line
- **C** (model comparison): Coarse (CLIP 8-class) vs 1000-way bars + pretrained model scatter
- **D** (data efficiency): Low-data regime (10K images) — coarse markers (2-64) + two dashed lines for 1000-way at 10K and 1.2M
- **E** (PC scatter): Behavioral ground truth, CNN 8-class, CNN 1K, ViT-B/16 1K, with super-category coloring + image insets

**Files:**
- `figure4.py` — main assembly
- `panel_coarseness.py` — Panel B
- `panel_comparison.py` — Panel C
- `panel_data_efficiency.py` — Panel D
- `panel_scatter.py` — Panel E data loading + image insets
- `plot_pc_scatter.py` — Panel E scatter plotting + super-category logic

**Data:** `results.db`, `experiments/coarse_grain_benefits/data_efficiency/legacy_results/data_efficiency_results.csv` (10K regime, not in DB), `pretrained_vit_things.npz`, THINGS images via `BONNER_DATASETS_HOME`.

## Figure 5: Per-Concept Alignment (`fig5/`)

Which individual concepts benefit from coarse vs fine-grained training.

**Layout:** Row 1: Three RDMs (Behavioral | 8-class CLIP | 1000-class) + colorbar + legend. Row 2: Per-concept scatter + advantage histogram.

**Panels:**
- **A**: Category-sorted RDMs — concepts grouped by 8 super-categories (from THINGS-27)
- **B**: Per-concept scatter — 8-class vs 1000-way per-concept RSA, colored by super-category
- **C**: Histogram of per-concept advantage (delta rho)

**Files:**
- `figure5.py` — full figure
- `extract_activations.py` — extracts FC1 activations for RDM computation

**Data:** `activations.npz`, `fc1_post_activations.npz`, `pretrained_alexnet_fc1.npz`, THINGS behavioral data via `bonner-datasets`.

## Figure 6: Architecture Generalization (`fig6/`)

Tests whether the coarseness advantage generalizes to standard architectures.

**Layout:** Single row, 3 panels: ResNet-50 | ConvNeXt | ViT-B/16. Each shows THINGS coarseness (CLIP labels) with coarse markers (2-64) and a 1000-class baseline bar.

**Files:**
- `figure6.py` — full figure

**Data:** `results.db` (THINGS results for ResNet-50, ConvNeXt, ViT-B/16).

## Supplementary Figures (`supplementary/`)

Supplementary figures live in `manuscript/figures/supplementary/` (NOT `manuscript/supplementary/`). Each figure lives in its own `SN_*/` subfolder containing the script and its output PNG(s) — mirroring the `figN/` layout of the main figures. See `supplementary/supplementary.md` for the full index and captions.

**Style:** Must match main Figure 3's scatter format exactly (same `panel_raw.py` approach). See `supplementary/figure_style.md` for the complete style reference.

**Key scripts:**
- `S1_coarsegrain_models/S1_coarsegrain_models.py` — S1A (neural: 2x2 TVSD+NSD) + S1B (THINGS behavioral), all 4 PCA sources
- `S2_encoding_scores/S2_encoding_scores.py` — 2x2 neural panel using encoding-score (RidgeCV Pearson r) instead of RSA; AlexNet + CLIP (no Pixels — absent from DB)
- `S3_wordnet/S3_wordnet.py` — WordNet-derived labels
- `S4_per_layer/S4_per_layer.py` — Full per-layer profiles
- `S5_finegrained_roi/S5_finegrained_roi.py` — Fine-grained ROI decomposition (NSD), raw RSA, AlexNet+CLIP
- `S6_pc_poles/S6_pc_poles.py` — PCA pole images
- `S7_reconstruction/S7_reconstruction.py` — Reconstruction control
- `S8_training_accuracy/S8_training_accuracy.py` — Training accuracy (scatter + 1000-way bar), all 4 PCA sources
- `S9_seed_variability/S9_seed_variability.py` — Seed variability

**Consistent style across S1, S2, S5, S8:** All use the same Figure 3 scatter format (broken x-axis, jittered markers, no connecting lines). Legend shows only PCA label sources — no 1000-way or untrained entries. Colors: AlexNet `#6baed6`, CLIP `#08519c`, ViT `#c0392b`, DINO `#1a8a7a`.
