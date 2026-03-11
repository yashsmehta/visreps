# Manuscript Figure Plan

**Status:** Working draft — figures are organized into 5 main figures with subdirectories (`fig1/`–`fig5/`). Supplementary figures listed at the end.

**PCA source models in main figures:** AlexNet, ViT, CLIP, Pixels. DINOv2 in supplementary only.

**Total models trained:** 7 granularity levels (2, 4, 8, 16, 32, 64, 1000) × 4 PCA sources (AlexNet, CLIP, ViT, Pixels) × 3 seeds + untrained controls = 84+ trained models from scratch, all on the same ~1.26M ImageNet images.

---

## Narrative Arc

The five main figures follow a progression: **method & expectations → representation analysis → human neural data (NSD) → macaque neural data (TVSD) → behavioral data (THINGS)**.

1. **Figure 1** introduces the coarse-graining procedure, shows the training setup, gives intuition for how representations change, and frames the conventional expectation.
2. **Figure 2** establishes that coarse-trained representations are genuinely different from 1000-way representations and cannot be recovered by low-dimensional projection of the fine-grained model.
3. **Figure 3** presents human fMRI alignment (NSD) with per-layer analysis and reconstruction controls.
4. **Figure 4** presents macaque electrophysiology alignment (TVSD) with the same structure.
5. **Figure 5** delivers the most surprising result: coarse models vastly outperform 1000-way on behavioral alignment (THINGS), with RDM visualizations and per-concept analysis explaining why.

---

## Figure 1: Method, Training Setup & Expectations

**Directory:** `manuscript/figures/fig1/`

**Narrative role:** Introduce the PCA-based coarse-graining procedure, show the training paradigm, give the reader visual intuition for how representations change, and set up the conventional expectation that fine-grained supervision should be necessary for brain-model alignment. No actual results yet — this figure is entirely about framing.

### Panel A — The coarse-graining procedure

Schematic of the pipeline: ImageNet images → pretrained source model (AlexNet, CLIP, ViT, or raw Pixels) → PCA on feature space (or pixel space) → recursive median splits along top PCs → coarse label sets (2, 4, 8, 16, 32, 64 classes).

The style should follow the spirit of the NeurIPS schematic (`experiments/neurips_2025/fig1/schematic_imagenet_pca.png`) — a clean, linear pipeline diagram — but freshly created for this manuscript with updated content. Show the flow from images to features to PCA projection to class assignments. Include example images at each split to give intuition for what PC1, PC2, etc. separate (e.g., PC1 roughly separates man-made vs. natural).

- *Key message:* The procedure is principled (data-driven, not arbitrary taxonomy), produces balanced classes, and uses only visual statistics — no linguistic or semantic labels.
- *Pixels variant:* Briefly note that one condition uses raw image pixels instead of pretrained features, as a baseline for minimal visual statistics.

**Existing assets:**
- `experiments/pca_visualization/figures/all_pcs_alexnet.png` — All PC poles (AlexNet features)
- `experiments/pca_visualization/figures/all_pcs_clip.png` — All PC poles (CLIP features)
- `experiments/pca_visualization/pc_histogram/` — PC score distributions
- `experiments/neurips_2025/fig1/schematic_imagenet_pca.png` — Prior NeurIPS schematic (reference style)
- `experiments/pca_visualization/generate_poles.py` — PC pole image generation
- `experiments/pca_visualization/visualize_poles.py` — Publication-quality pole layouts
- `experiments/pca_visualization/pc_semantic_analysis.py` — Semantic enrichment of PC poles

### Panel B — DNN training schematic

Minimal diagram showing the training paradigm: same architecture (CustomCNN / AlexNet-style) × same ~1.26M ImageNet images × same training protocol (SGD, 20 epochs, same augmentations), only varying the output classification head (2, 4, 8, 16, 32, 64, or 1000 classes). Three seeds per condition. Labels from four PCA source models (AlexNet, CLIP, ViT, Pixels).

- *Key message:* The **only** variable is label granularity. Architecture, data, and training procedure are identical across all conditions. This is a clean, controlled experiment with 84+ models trained from scratch.
- *Visual elements:* A schematic CNN diagram with the final FC layer branching into different output sizes. Keep it simple — no table or grid, just the schematic emphasizing the controlled design.

### Panel C — PC1/PC2 visualizations: how representations change

Side-by-side PC1 vs. PC2 scatter plots of ImageNet activations for a coarse-trained model vs. the 1000-way model. Points colored by 4-way PCA label.

- *Layer choice:* Compare FC1 and FC2 and pick whichever shows the more stark visual difference. FC2 may show the clearest divergence (coarse models collapse to class clusters in FC2 while 1000-way has a smooth gradient), but FC1 might show richer intermediate structure. Generate both and pick the most compelling.
- *Key visual:* The 1000-way model has a smooth gradient in PC space; the coarse model has clearly structured clusters that are internally variable. Different geometry, not just reduced geometry.
- *Key message:* Give the reader immediate visual intuition that coarse training fundamentally changes how the network organizes visual information, before showing any quantitative results.

**Existing assets:**
- `experiments/representation_analysis/2pcs_compare/pc_quadrant_pretrained_vs_4way_fc1.png` — FC1 comparison
- `experiments/representation_analysis/2pcs_compare/pc_quadrant_pretrained_vs_4way_fc2.png` — FC2 comparison
- `experiments/representation_analysis/2pcs_compare/pc_quadrant_pretrained_vs_4way_conv4.png` — conv4 comparison (backup)
- `experiments/representation_analysis/2pcs_compare/run_analysis.py` — Feature extraction + PCA + plotting
- `experiments/representation_analysis/2pcs_compare/plot.py` — Visualization

### Panel D — Conventional expectations (setting the premise)

A conceptual/schematic plot showing what one *might* expect for the relationship between label granularity and brain/behavioral alignment. This is a thought-experiment panel — no real data.

Two evaluation paradigms shown with icons:
1. **Brain alignment** (brain icon — fMRI/electrode schematic): Number of classes (x-axis, log scale) vs. neural alignment (y-axis).
2. **Behavioral alignment** (human/behavior icon — triplet task schematic): Same axes.

The conventional expectation could take several forms:
- *Monotonic increase:* More categories → richer supervision → more brain-like features. The "fine-grained supervision hypothesis" — the default in the field.
- *Flat-then-jump:* Alignment is low for coarse labels, then jumps once you have enough classes to force the network to learn fine-grained visual features (e.g., for object classification).
- *Diminishing returns:* Rapid initial gains, then saturation — but always increasing.

All these share the common assumption that **1000-way should be best or at least near-best**. Show one or two of these as schematic curves (the monotonic increase being the primary expectation) with a question mark: "Is fine-grained supervision necessary?"

- *Key message:* Set up the conventional wisdom as the null hypothesis. No dataset details or schematics here — just the conceptual framing. The actual datasets are introduced in Figures 3–5. Do NOT reveal the actual pattern — let the data deliver the surprise.

**Design note:** Consider showing the two evaluation types (brain, behavior) as separate small schematic plots side by side, since the actual results will diverge between them (brain: coarse ≈ 1000-way; behavior: coarse >> 1000-way). This foreshadows without spoiling.

---

## Figure 2: Coarse Representations Are Fundamentally Different

**Directory:** `manuscript/figures/fig2/`

**Narrative role:** Establish that the internal representations learned from coarse supervision are *qualitatively distinct* from 1000-way representations. They are not just a low-dimensional projection or subset of the fine-grained features. This figure answers two key questions: (1) How different are coarse representations from fine-grained? (2) Can you recover them by simple dimensionality reduction of the 1000-way model? The answer to the second question is no — and this matters because it means the brain alignment results in Figures 3–5 cannot be dismissed as a trivial dimensionality artifact.

### Panel A — Cross-model RSA: coarse ≠ fine-grained

Adapted from the NeurIPS paper (Figure 1). Compute RSA between internal representations of different models:

1. **Cross-seed baseline:** RSA between two 1000-way models trained with different seeds. This establishes the expected level of representational similarity between models that are nominally identical (same training, different initialization).
2. **1000-way vs. coarse:** RSA between a 1000-way model and a coarse model (e.g., 2-way, 4-way, ..., 64-way). If coarse models simply learned a subset of the same features, this should approach the cross-seed baseline. Instead, it is substantially lower — especially at low granularity.

- *Key visual:* Bar plot or line plot. X-axis: coarseness level (2, 4, 8, ..., 64). Y-axis: model-to-model RSA. Horizontal dashed line at the cross-seed baseline. Coarse models fall well below.
- *Key message:* Coarse training produces representations that are not merely "less" than fine-grained — they are *different*.

**Existing assets:**
- `experiments/neurips_2025/fig1/model_reps_rsa_comparisons.py` — Cross-model RSA (inter-seed, 1K-vs-coarse, reconstructed-vs-coarse)
- `experiments/neurips_2025/fig1/rsa_comparison_pc2_Kendall_bars.png` — Cross-seed and cross-model RSA bars

### Panel B — Low-dimensional projection cannot recover coarse representations

Directly addresses: "Are coarse representations just the top-k PCs of the 1000-way model?"

For each coarseness level, project the 1000-way model's FC1 activations onto the top-k PCs, where **k = log₂(n_classes)** — i.e., matching the number of PCA splits used to generate the coarse labels:
- 2 classes → top-1 PC
- 4 classes → top-2 PCs
- 8 classes → top-3 PCs
- 16 classes → top-4 PCs
- 32 classes → top-5 PCs
- 64 classes → top-6 PCs

Then compute RSA between the projected 1000-way FC1 representations and the actual coarse model's FC1 representations.

- *X-axis:* Number of classes (2, 4, 8, 16, 32, 64), log₂ scale
- *Y-axis:* Model-to-model RSA (Spearman or Kendall)
- *Key visual:* Line or bar plot showing that projected-1000-way vs. coarse RSA remains low across all granularity levels. Even when projecting onto the exact PCs that define the label space, the 1000-way model's activations in that subspace do not resemble what the coarse model actually learns.
- *Key message:* You cannot get coarse representations by dimensionality reduction of fine-grained representations. The coarse training signal drives the network to discover a different set of visual features — not just the projections that define class boundaries.

**Existing assets:**
- `experiments/neurips_2025/fig1/model_reps_rsa_comparisons.py` — Cross-model RSA (inter-seed, 1K-vs-coarse, reconstructed-vs-coarse)
- `experiments/reconstruction_analysis/` — PCA reconstruction framework

### ~~Panel C — Eigenspectrum and effective dimensionality~~ (REMOVED)

**Removed from Figure 2.** Eigenspectra, participation ratio, intrinsic dimensionality, and sparsity analyses are now covered in **Supplementary Figure S9** instead. Figure 2 focuses exclusively on cross-model RSA (Panel A) and the projection control (Panel B) — keeping it tight on the "coarse ≠ fine-grained" message without overloading with representational geometry details.

---

## Figure 3: Human fMRI Alignment (NSD)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present the core human neural data. Coarse-trained models match or approach 1000-way models in brain alignment across cortical regions. Include per-layer analysis and dual PCA reconstruction controls. This is the first of two neural datasets — convergent results across species (Fig 3 + Fig 4) make the findings especially compelling.

### Panel A — NSD dataset schematic

Brief visual description of the NSD dataset:
- Human brain with fMRI overlay
- 8 subjects, 7T fMRI, ~10K natural scene images
- Two region groupings: early visual stream (V1–V3) and ventral visual stream (V4, FFA, PPA)
- RSA methodology: compare model RDMs to neural RDMs

- *Visual elements:* Small brain diagram highlighting the two streams. Minimal — the reader should grasp the dataset in 5 seconds.
- *Key message:* High-quality human fMRI data with multiple subjects and well-defined cortical regions.

### Layout — 2×3 grid

Figure 3 is a **2 row × 3 column** grid. Rows are regions, columns are analyses:

|  | **Col 1: Coarseness** | **Col 2: Per-layer** | **Col 3: Reconstruction** |
|--|----------------------|---------------------|--------------------------|
| **Row 1:** Early visual stream | Alignment vs. # classes | RSA across layers | Dual PCA curves |
| **Row 2:** Ventral visual stream | Alignment vs. # classes | RSA across layers | Dual PCA curves |

### Column 1 — Coarseness results (log-scale)

Alignment (Spearman ρ) vs. number of classes (log₂ x-axis, 2 → 1000). Four PCA source models (AlexNet, CLIP, ViT, Pixels) as separate markers/colors, plus 1000-way baseline (diamond) and untrained baseline (horizontal dashed line).

- *Key pattern — Early visual stream (top):* Flat or slightly increasing from 2 → 64, all coarse models saturate early (~8–16 classes). Coarse models match or exceed 1000-way. Pixels PCA sits well below (captures only low-level correlations).
- *Key pattern — Ventral visual stream (bottom):* Gradual increase from 2 → 64 classes, approaching or matching 1000-way. Even here, 32–64 classes capture most of the alignment. Pixels PCA again much lower.
- *Error bars:* Bootstrap CIs aggregated across 8 subjects × 3 seeds.
- *Architectures:* AlexNet (blue circle), CLIP (green square), ViT (magenta triangle-up), Pixels (orange triangle-down). No DINO in main figures.

**Existing assets:**
- `manuscript/figures/plot_coarseness_log.py` — Current script (already includes Pixels)
- `manuscript/figures/coarseness_log.png` — Current output

### Column 2 — Per-layer analysis

For a selected PCA architecture (e.g., CLIP), show RSA scores across all network layers for multiple granularity levels. **Average across all 8 subjects — no error bars or spread** (keeps it clean; per-subject breakdowns are in Supplementary S4).

- *X-axis:* Network layers — **14 data points** per model: `conv1_pre`, `conv1_post`, `conv2_pre`, `conv2_post`, ..., `conv5_post`, `fc1_pre`, `fc1_post`, `fc2_pre`, `fc2_post`. The `_pre` taps are before BatchNorm+ReLU, `_post` taps are after. This gives a fine-grained view of how alignment evolves through the network, including across nonlinearities.
- *Y-axis:* Spearman ρ
- *Lines:* One per granularity (2, 4, 8, 16, 32, 64, 1000), colored by the blue gradient + distinct color for 1000-way
- *Top panel:* Early visual stream
- *Bottom panel:* Ventral visual stream
- *Key pattern:* Best layer may shift from deeper (fc) layers for fine-grained to more intermediate (conv4–conv5) layers for coarse. The coarse models' peak alignment at intermediate layers suggests they develop general-purpose features rather than task-specific features.
- *Note:* Could alternatively use CLIP as the architecture, since CLIP-PCA labels produce the strongest ventral stream alignment at intermediate granularity.

**Data source:** `layer_selection_scores` table in `results.db` (30,086 rows, ~14 layers per run — `conv1-5_pre/post`, `fc1-2_pre/post`).

**Existing assets:**
- `results.db` → `layer_selection_scores` table
- Plotting: needs new script

### Column 3 — Dual PCA reconstruction control

Can you achieve the same brain alignment by taking the top-k PCs of the 1000-way model? What about the top-k PCs of the coarse model itself? Each sub-panel plots **two reconstruction curves**:

1. **1000-way reconstruction** (golden amber curve): Take the 1000-way model's best-layer activations, retain only the top-k PCs (k = 1, 2, 3, ..., 15), and compute brain alignment.
2. **Coarse model reconstruction** (blue curve): Same procedure but for the best coarse model (region-specific: AlexNet-PCA 64-way for early visual stream, CLIP-PCA 16-way for ventral visual stream).

Plus reference lines:
- **Untrained baseline** (gray dotted): alignment of an untrained network

- *Top panel:* Early visual stream
- *Bottom panel:* Ventral visual stream
- *Key pattern:* The coarse model reconstruction curve should sit above or near the 1000-way curve — its top PCs are more brain-aligned per PC. The 1000-way model's top PCs plateau below the coarse model's, showing that dimensionality reduction of the fine-grained model cannot recover the coarse model's alignment.
- *Key message:* The coarse model's brain alignment is not an artifact of low dimensionality. Its principal components capture genuinely different (and more brain-relevant) visual features than the 1000-way model's top PCs.

**Existing assets:**
- `experiments/reconstruction_analysis/plot.py` — Dual-curve plotting (calls `plot_dual_figure`)
- `experiments/reconstruction_analysis/plot_utils.py` — `plot_dual_curves()` function (draws both curves)
- `experiments/reconstruction_analysis/figures/reconstruction_nsd.png` — Current NSD dual reconstruction figure
- `results.db` → `reconstruct_from_pcs=1` rows: 1000-way (cfg_id=1000, imagenet1k) AND coarse models (e.g., cfg_id=64/pca_labels_alexnet for early, cfg_id=16/pca_labels_clip for ventral)

**Reconstruction data in results.db:**
- NSD 1000-way: 34 condition rows (pca_k × region)
- NSD coarse (AlexNet 64-way): 17 rows; (CLIP 16-way): 17 rows

---

## Figure 4: Macaque Electrophysiology Alignment (TVSD)

**Directory:** `manuscript/figures/fig4/`

**Narrative role:** Cross-species validation. The same pattern observed in human fMRI (Figure 3) holds in macaque spiking data. This rules out measurement-modality-specific artifacts and strengthens the claim that coarse-grained features are genuinely brain-like. **V4 is omitted from the main figure** to keep the layout parallel with Figure 3 (2 rows) and because V1 and IT represent the two extremes of the visual hierarchy — V4 results are available in supplementary.

### Panel A — TVSD dataset schematic

Brief visual description of the TVSD dataset:
- Macaque brain with electrode array illustration
- 2 monkeys, Utah arrays in V1, V4, IT
- ~22K images, multi-unit spiking activity
- Same RSA methodology as NSD

- *Visual elements:* Macaque brain diagram with electrode positions. Minimal.

### Layout — 2×3 grid (parallel to Figure 3)

|  | **Col 1: Coarseness** | **Col 2: Per-layer** | **Col 3: Reconstruction** |
|--|----------------------|---------------------|--------------------------|
| **Row 1:** V1 | Alignment vs. # classes | RSA across layers | Dual PCA curves |
| **Row 2:** IT | Alignment vs. # classes | RSA across layers | Dual PCA curves |

V4 results moved to supplementary (see S-figure note below).

### Column 1 — Coarseness results (log-scale)

Same format as Figure 3 Column 1. Two sub-panels: V1 (top), IT (bottom).

- *Key pattern — V1 (top):* Coarse models match or outperform 1000-way (parallels NSD early visual stream).
- *Key pattern — IT (bottom):* Gradual increase, 32–64 classes approach 1000-way (parallels NSD ventral visual stream).
- *Error bars:* ±1.96 SEM across 2 monkeys × 3 seeds (wider CIs due to fewer subjects).
- *Architectures:* AlexNet, CLIP, ViT, Pixels (once data available; currently only cfg_id=2 for Pixels).

**Existing assets:**
- `manuscript/figures/plot_coarseness_log_tvsd.py` — Current script (needs Pixels addition when data is ready)
- `manuscript/figures/coarseness_log_tvsd.png` — Current output

### Column 2 — Per-layer analysis

Same format as Figure 3 Column 2. **Average across 2 monkeys — no error bars or spread.** One panel per region (V1 top, IT bottom). All 14 layer taps (`conv1-5_pre/post`, `fc1-2_pre/post`) on x-axis.

- *Lines:* One per granularity (2, 4, 8, 16, 32, 64, 1000), colored by the blue gradient + distinct color for 1000-way
- *Key pattern:* In IT, best layer shifts with granularity, similar to NSD ventral stream. In V1, per-layer profile is flatter across granularity levels.

**Data source:** `layer_selection_scores` in `results.db`, filtered for `neural_dataset='tvsd'`.

### Column 3 — Dual PCA reconstruction control

Same dual-curve format as Figure 3 Column 3. Both the 1000-way and coarse model (AlexNet-PCA 64-way) reconstruction curves. Two sub-panels: V1 (top), IT (bottom).

**Existing assets:**
- `experiments/reconstruction_analysis/plot.py` — Same script handles TVSD
- `experiments/reconstruction_analysis/figures/reconstruction_tvsd.png` — Current TVSD dual reconstruction figure
- `results.db` → `reconstruct_from_pcs=1` rows: 1000-way (51 rows) and AlexNet-PCA 64-way (51 rows) for TVSD

**Note on V4:** V4 coarseness, per-layer, and reconstruction results should be included as a supplementary figure for completeness.

---

## Figure 5: Behavioral Alignment (THINGS) — The Surprise

**Directory:** `manuscript/figures/fig5/`

**Narrative role:** Deliver the most surprising and impactful result: coarse-trained models *vastly* outperform the 1000-way model on behavioral alignment. This is not a subtle effect — the gap is ~0.1–0.18 in Spearman ρ, with even 2-class models beating 1000-way. Then explain *why* through RDM visualizations and per-concept analysis. This figure transitions from result to mechanism and is the climax of the paper.

### Panel A — THINGS dataset schematic

Brief visual description of the THINGS dataset:
- 1,854 everyday object concepts
- Human similarity judgments from odd-one-out triplet task
- 80/20 concept-level train/test split (seed=42)
- No regions, no per-subject structure — single aggregate similarity matrix

- *Visual elements:* Triplet task illustration (three images, "which is the odd one out?"). Minimal.

### Panel B — Coarseness bar plots

The headline result. Single-panel bar plot per PCA architecture: Untrained (gray) | coarse 2–64 (blue gradient, hatched) | break marks | 1000-way (orange).

- *Key pattern:* All coarse models (2–64 classes) cluster around ρ ≈ 0.47–0.58, while the 1000-way model drops to ~0.40. Even 2-class models substantially outperform 1000-way. Fine-grained supervision actively *hurts* behavioral alignment.
- *Error bars:* Bootstrap 95% CIs across 3 seeds.
- *One figure per architecture:* `--pca_labels alexnet`, `--pca_labels clip`, etc. The main figure should show the architecture with the most striking effect (likely CLIP), with others in supplementary or as sub-panels.

**Existing assets:**
- `manuscript/figures/plot_coarseness_things.py` — Bar plot script (uses `plot_helpers.plot_coarseness_bars`)
- `manuscript/figures/coarseness_bars_alexnet.png` — Current AlexNet output

### Panel C — Category-annotated RDMs

The most visually striking evidence for *why* coarse models win. Three-panel RDM comparison (no difference RDM — the visual contrast between the three panels speaks for itself):

1. **Human behavioral RDM** — ground truth similarity structure from THINGS triplet judgments
2. **Coarse model RDM** (e.g., CLIP 4-class) — captures the broad block structure
3. **1000-way model RDM** — imposes finer distinctions that don't match human judgments

Concepts sorted by the 27 THINGS semantic categories with boundary lines overlaid.

- *Key visual:* The coarse model RDM captures the broad categorical block structure of human similarity (animals grouped together, food grouped together, vehicles grouped together) much better than the 1000-way model, which over-differentiates within categories.
- *Key message:* Fine-grained training forces the network to emphasize within-category distinctions (needed to tell apart 1000 classes) at the expense of the broad between-category structure that dominates human similarity judgments.

**Existing assets:**
- `experiments/things_visualizations/plot_rdms_categorized.py` — RDM with semantic category boundaries
- `experiments/things_visualizations/figures/rdm_categorized.png` — Current categorized RDM output
- `experiments/things_visualizations/plot_rdms.py` — Per-layer RDMs

### Panel D — Per-concept scatter: which concepts drive the advantage?

Scatter plot of per-concept RSA contribution: coarse model (e.g., CLIP 4-class, y-axis) vs. 1000-way model (x-axis). Identity line for reference. Points color-coded by THINGS semantic category. Optional marginal histogram of differences.

- *Key pattern:* ~70% of concepts fall above the diagonal (coarse model wins). The advantage is broad and systematic, not driven by outliers. Plants, animals, clothing accessories strongly favor coarse. Body parts, drinks favor 1000-way.
- *Key message:* The coarse model advantage is pervasive across most concept categories, not a niche effect.

**Existing assets:**
- `experiments/things_visualizations/per_row_scatter_categories.py` — Per-concept scatter colored by category
- `experiments/things_visualizations/per_row_rdm.py` — Per-concept RDM analysis
- `experiments/things_visualizations/figures/per_row_scatter_categories.png` — Current scatter output
- `experiments/things_visualizations/figures/per_row_categories.png` — Category-colored version

### Panel E — Dual PCA reconstruction control (THINGS)

Same dual-curve format as Figures 3D/4D. Both the 1000-way and coarse model (ViT-PCA 64-way) reconstruction curves. The gap here is the most dramatic of any benchmark.

- *Key pattern:* Best coarse model (ρ ≈ 0.58) far exceeds both the 1000-way top-k PCs curve (plateaus ~0.39) AND the coarse model's own top-k PCs curve (which rises steeply but from the same direction). The full coarse model captures behavioral similarity that is distributed across many of its dimensions — not just concentrated in the top few PCs.
- *Key message:* The coarse model's behavioral advantage cannot be explained by dimensionality reduction of either model. It genuinely learns different features.

**Existing assets:**
- `experiments/reconstruction_analysis/plot.py` — Same script handles THINGS
- `experiments/reconstruction_analysis/figures/reconstruction_things-behavior.png` — Current THINGS dual reconstruction figure
- `results.db` → `reconstruct_from_pcs=1` rows: 1000-way (17 rows) and ViT-PCA 64-way (17 rows) for THINGS

### Panel F (optional) — Semantic dimension profiling

Horizontal bar chart: correlation of per-concept advantage (coarse minus 1000-way) with each of the 66 THINGS behavioral dimensions. Reveals which perceptual dimensions drive the coarse model advantage.

- *Key pattern:* "Animal-related," "plant-related," "house/furnishing-related" show strong coarse advantage. "Body-part-related," "fluffy/soft" show 1000-way advantage.
- *Key message:* The coarse model better captures *high-variance* perceptual dimensions that structure human similarity (animacy, naturalness), while 1000-way over-specializes for class identity.

**Existing assets:**
- `experiments/things_visualizations/plot_dimension_alignment.py` — Dimension alignment analysis
- `experiments/things_visualizations/figures/dimension_profiling.png`
- `experiments/things_visualizations/figures/dimension_alignment.png`
- `experiments/things_visualizations/figures/dimension_difference.png`

### Panel G (optional) — Image collages

Representative images for top concepts where coarse wins vs. where 1000-way wins. Immediate visual intuition.

**Existing assets:**
- `experiments/things_visualizations/figures/collage_clip4_wins.png`
- `experiments/things_visualizations/figures/collage_1k_wins.png`
- `experiments/things_visualizations/characterize_discrepant.py` — Analysis of discrepant concepts

---

## Supplementary Figures (Candidates)

### S1. Training summary, model inventory & coarse-class accuracy
Comprehensive overview of all models trained: granularity levels × PCA source models × seeds, training convergence curves (loss, accuracy over epochs), and **final classification accuracy for each granularity level**. The accuracy panel is important: it confirms that models actually learn their respective coarse classification tasks (e.g., 2-class models reach ~99% accuracy, 64-class models ~85%). Without this, a reviewer might wonder whether the coarse models are simply undertrained or failing to learn. Show accuracy as a function of number of classes (log x-axis) with error bars across seeds.
- Communicates the scale and rigor of the experiment — 84+ models trained from scratch under identical conditions.

### S2. DINOv2 PCA source model results
All main figure analyses repeated with DINOv2 labels (NSD only; limited coverage). Verifies that the main findings are not specific to any single PCA source model.

### S3. NSD-Synthetic (OOD) results
Coarse models maintain alignment on synthetic/out-of-distribution stimuli. Same log-scale format as Figure 3B.
- `plotters/nsd_synthetic/plot_coarseness.py`

### S4. Per-subject NSD breakdowns
Dot/box plots showing consistency across all 8 NSD subjects for both early and ventral visual streams. Demonstrates that the main effects are not driven by outlier subjects.
- `plotters/nsd/` per-subject figures

### S5. Fine-grained ROI analysis (NSD)
V1, V2, V3, hV4, FFA, PPA individually for NSD. Shows the coarseness effect at finer anatomical resolution.
- `plotters/nsd/plot_coarseness.py --regions finegrained`

### S6. Stimulus robustness
RSA stability under stimulus subsampling. Shows that brain alignment estimates are robust to which specific stimuli are included.
- `experiments/stimulus_robustness/`

### S7. Stimulus sensitivity
k-fold CV RSA fluctuation analysis. Quantifies variability in alignment estimates across cross-validation folds.
- `experiments/stimulus_sensitivity/`

### S8. Pre- vs. post-ReLU comparison
Dimensionality metrics (eigenspectrum, participation ratio, intrinsic dimension, sparsity) before and after ReLU activation. Shows how the nonlinearity reshapes the representational geometry.
- `experiments/representation_analysis/pre_post_relu_comparison.py`
- `experiments/representation_analysis/figs/pre_post_relu_comparison.png`

### S9. Internal representational analysis across granularity levels
Full suite of representational geometry metrics as a function of label granularity (2, 4, 8, 16, 32, 64, 1000) and network layer. Four panels:

1. **Eigenspectrum:** Log-log plot of eigenvalues for each granularity level at a selected layer (e.g., fc1). Coarse models concentrate variance in fewer dimensions (steeper falloff). Shows how the spectral structure of representations changes with supervision granularity.
2. **Effective dimensionality (participation ratio):** Across layers (x-axis: conv1 → fc2) for each granularity level. Coarse models have lower participation ratios, especially in later layers. Quantifies how many dimensions carry meaningful variance.
3. **Intrinsic dimensionality:** Two-nearest-neighbor or MLE estimator across layers. Complements participation ratio with a nonlinear dimensionality measure — captures manifold complexity, not just spectral spread.
4. **Sparsity (lifetime sparsity / kurtosis):** Across layers for each granularity level. Coarse models may develop sparser, more interpretable feature maps. Shows whether coarse training produces more selective unit responses.

- *Key message:* Coarse training produces representations that are lower-dimensional, spectrally concentrated, and potentially sparser — providing the geometric basis for why these representations align differently with brain and behavioral data.
- *Design:* 2×2 grid (eigenspectrum, participation ratio, intrinsic dim, sparsity), one line per granularity level using the blue gradient + distinct color for 1000-way. Could show a single PCA source (e.g., AlexNet) in main supplementary, others available on request.

**Existing assets:**
- `experiments/representation_analysis/representation_summary.py` — 2×2 summary (generates all four panels)
- `experiments/representation_analysis/figs/representation_summary.png` — Current 2×2 summary figure
- `experiments/representation_analysis/figs/representation_summary_data.json` — Cached data
- `experiments/representation_analysis/dimensionality/eigenspectrum.png` — Individual eigenspectrum
- `experiments/representation_analysis/dimensionality/participation_ratio.png` — Individual participation ratio
- `experiments/representation_analysis/dimensionality/intrinsic_dimension.png` — Individual intrinsic dim
- `experiments/representation_analysis/dimensionality/sparsity.png` — Individual sparsity
- `experiments/representation_analysis/dimensionality/metrics.py` — Metric computation
- `experiments/representation_analysis/dimensionality/plots.py` — Plotting functions
- `experiments/representation_analysis/dimensionality/run.py` — Runner script

### S10. Additional THINGS architectures
Bar plots for each PCA source model (AlexNet, CLIP, ViT, Pixels) as individual supplementary panels. The main figure shows only the most striking architecture; this supplements with all others.

### S11. Nearest neighbor retrieval
Same query image, different neighbors retrieved from coarse vs. fine-grained model feature spaces. Qualitative illustration of how coarse models organize images differently — coarse retrieves perceptually similar but semantically diverse images, while 1000-way retrieves same-class images.
- `experiments/representation_analysis/nearest_neighbors.py`

### S12. PC axis interpretation: what do the principal components correspond to?
Most-activating and least-activating ImageNet images for the top principal components of each PCA source model (AlexNet, CLIP, ViT, Pixels). Gives the reader intuition for what the axes along which models are being trained actually represent — PC1 separates man-made vs. natural, PC2 captures scene structure vs. small objects, etc.
- `experiments/pca_visualization/generate_poles.py` — Generate rankings
- `experiments/pca_visualization/visualize_poles.py` — Publication-quality pole images
- `experiments/pca_visualization/figures/all_pcs_alexnet.png`, `all_pcs_clip.png`
- `experiments/pca_visualization/pc_semantic_analysis.py` — Semantic enrichment analysis
- `experiments/model_activating_images/` — Most-activating images per output class

### S13. Levels evaluation (tentative)
Granularity levels dataset results. Status uncertain — include if results are compelling.
- `experiments/levels_evaluation/`

---

## Directory Structure

```
manuscript/figures/
├── figure_plan.md              # This file
├── fig1/                       # Method, setup, expectations
│   └── (schematics, PC space visualizations, conceptual plots)
├── fig2/                       # Representations are different
│   └── (cross-model RSA, projection analysis, eigenspectra)
├── fig3/                       # NSD results
│   ├── plot_coarseness_log.py  # Log-scale coarseness (moved from root)
│   └── (per-layer, dual reconstruction)
├── fig4/                       # TVSD results
│   ├── plot_coarseness_log_tvsd.py  # Log-scale coarseness (moved from root)
│   └── (per-layer, dual reconstruction)
├── fig5/                       # THINGS behavioral results
│   ├── plot_coarseness_things.py    # Bar plot (moved from root)
│   └── (RDMs, scatter, dual reconstruction, dimension profiling)
├── plot_coarseness_log.py      # (legacy location, to be moved to fig3/)
├── plot_coarseness_log_tvsd.py # (legacy location, to be moved to fig4/)
└── plot_coarseness_things.py   # (legacy location, to be moved to fig5/)
```

Scripts will be migrated into subdirectories as panels are finalized. Current scripts remain at the root level until the reorganization is complete.

---

## Design Notes

- **Consistent color scheme across all figures:**
  - PCA source models: AlexNet (blue, `#2166AC`, circle), CLIP (green, `#1B7837`, square), ViT (magenta, `#C51B7D`, triangle-up), Pixels (orange, `#E08214`, triangle-down)
  - Granularity levels: Blue gradient (light → dark for 2 → 64)
  - 1000-way baseline: Dark gray (`#404040`, diamond)
  - Untrained baseline: Light gray dashed line (`#AAAAAA`)
  - Reconstruction curves: Golden amber (`#e6a200`) for 1000-way, blue (`#2166AC`) for coarse model
- **Log-scale x-axis** for all coarseness scatter/line plots (Figures 3B, 4B). THINGS uses bar plots instead (Figure 5B).
- **Schematics should be simple** — the reader should grasp each dataset in 5–10 seconds.
- **Reconstruction controls inline with dual curves** — placed within the same figure as the results they validate (Figures 3D, 4D, 5E). Each shows both the 1000-way and coarse model reconstruction trajectories, not just the 1000-way curve with a flat coarse baseline.
- **Figure 5 is the climax** — give it the most space. The behavioral result is the most surprising and needs the most explanation (RDMs, per-concept analysis, dimension profiling).
- **No DINO in main figures** — supplementary only (limited coverage).
- **No NSD-Synthetic in main figures** — supplementary only.
- **No encoding score or downstream benefits in main or supplementary** — the paper's focus is brain/behavioral alignment, not downstream task performance.
- **Pixels PCA status:** Full NSD data available (cfg_id 2–64). THINGS and TVSD experiments in progress (only cfg_id=2 so far). Plots will be updated as data becomes available.

---

## Data Availability Tracker

| Dataset | AlexNet | CLIP | ViT | Pixels | Notes |
|---------|---------|------|-----|--------|-------|
| NSD (early + ventral) | Full (2–64, 3 seeds) | Full | Full | Full (2–64, 3 seeds) | Ready to plot |
| TVSD (V1, V4, IT) | Full | Full | Full | cfg_id=2 only (2 seeds) | Experiments running |
| THINGS-behavior | Full | Full | Full | cfg_id=2 only (2 seeds) | Experiments running |
| NSD-Synthetic | Full | Full | Partial | — | Supplementary only |

### Reconstruction data coverage (reconstruct_from_pcs=1)

| Dataset | 1000-way | Coarse model | Notes |
|---------|----------|--------------|-------|
| NSD | 34 rows (pca_k × region) | AlexNet 64-way (17), CLIP 16-way (17) | Region-specific coarse models |
| TVSD | 51 rows (pca_k × region × subject) | AlexNet 64-way (51) | All three regions |
| THINGS | 17 rows (pca_k) | ViT 64-way (17) | Single panel |
