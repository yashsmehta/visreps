# Figure Plan

**Status:** Working draft — Figure 2 panels pending updated representation analysis results.

## Narrative Arc

The four main figures follow a progression: **method → representation → neural evidence → behavioral evidence**.

1. Figure 1 introduces the coarse-graining procedure, gives intuition for the classes, and frames the conventional expectation.
2. Figure 2 establishes that coarse-trained representations are genuinely different and interesting.
3. Figure 3 presents neural alignment (NSD + TVSD) with inline reconstruction controls.
4. Figure 4 delivers the surprising behavioral results (THINGS) with inline reconstruction control and mechanistic explanation.

**PCA source models in main figures:** AlexNet, ViT, CLIP. DINOv2 in supplementary only.

---

## Figure 1: Method Schematic and Expectations

**Narrative role:** Introduce the PCA-based coarse-graining procedure, give the reader intuition for what the coarse classes look like, and frame the conventional expectation: fine-grained supervision should be necessary for brain-model alignment.

### Panels

**Panel A — The coarse-graining procedure.**
Schematic of the pipeline: ImageNet images → pretrained source model → PCA on feature space → recursive median splits along top PCs → coarse label sets (2, 4, 8, ..., 64 classes). Show visually how the first PC splits all images into two groups, the second PC further subdivides (yielding 4 classes), and so on.

- *Visual elements:* Tree/dendrogram-like diagram showing recursive binary splits. At each split, a few representative images on each side. Color-code groups consistently with later figures.
- *Key message:* The procedure is principled (data-driven, not arbitrary taxonomy), produces balanced classes, and uses only visual statistics — no linguistic or semantic labels.

**Panel B — What do the coarse classes look like?**
PC pole visualizations: for each of the top 6 PCs, show exemplar images at the positive and negative poles. PC1 roughly separates man-made vs. natural objects, PC2 captures outdoor scenes vs. small round objects, etc. — these are *emergent* visual distinctions, not pre-defined semantic categories.

- *Data source:* `experiments/pca_visualization/figures/all_pcs_alexnet.png`
- *Key message:* The coarse labels capture broad, visually coherent groupings that are interpretable but not aligned to any human-defined taxonomy.

**Panel C — The conventional expectation.**
A conceptual plot: number of classes (x-axis, log scale) vs. brain-model alignment (y-axis). Show a single monotonic increasing curve representing the field's default assumption: more categories → richer supervision → more brain-like features. This is the "fine-grained supervision hypothesis" — the prevailing view that 1000-way ImageNet classification is the gold standard because rich category structure forces networks to learn detailed visual features.

- *Visual elements:* One clean schematic curve (monotonic increase), with a question mark or annotation posing: "Is fine-grained supervision necessary?"
- *Key message:* Set up the conventional wisdom as the null hypothesis. Do NOT reveal the actual pattern — let the data in Figures 3 and 4 deliver the surprise.

**Panel D — Training setup (compact).**
Minimal diagram: same architecture (AlexNet) × same ~1.26M images × same training protocol, only varying the output head (2, 4, ..., 64, 1000 classes). Three seeds per condition. Labels come from three different source models (AlexNet, CLIP, ViT). Emphasize: the *only* variable is label granularity.

### Existing assets
- `experiments/pca_visualization/figures/all_pcs_alexnet.png` — PC pole images
- `experiments/pca_visualization/pc_histogram/` — PC score distributions
- `experiments/neurips_2025/fig1/schematic_imagenet_pca.png` — prior NeurIPS schematic
- `experiments/pca_visualization/pc_semantic_analysis.py` — semantic enrichment of PC poles

---

## Figure 2: Coarse-Trained Representations Are Genuinely Different

**Narrative role:** Show that the internal representations learned from coarse supervision are not just low-dimensional projections of a fine-grained model's features — they are qualitatively distinct. The reader should come away understanding that these models organize visual information in a fundamentally different way.

**Status: Panel content pending updated results** for sparsity, effective dimensionality, and eigenspectra. The overall structure and candidate panels are outlined below.

### Candidate panels

**Panel A — PC space visualization: 1000-way vs. coarse-trained.**
Side-by-side PC1/PC2 scatter plots of ImageNet activations. Points colored by 4-way PCA label. Best shown at conv4 (where the coarse model still has rich, spread-out structure, unlike fc2 where everything collapses to a point) — this conveys that the model develops interesting features *despite* minimal supervision.

- *Data source:* `experiments/representation_analysis/2pcs_compare/`
- *Key visual:* The 1000-way model has a smooth gradient; the coarse model has clearly structured clusters that are internally variable. Different geometry, not just reduced geometry.

**Panel B — Eigenspectrum and effective dimensionality.**
Compare eigenvalue spectra and effective dimensionality (participation ratio) across layers for 1000-way vs. coarse models. The coarse model concentrates variance in fewer dimensions (steeper eigenspectrum, lower participation ratio). This sets up the reconstruction control in Figures 3–4: despite being lower-dimensional, these dimensions are *not* the same as the 1000-way model's top PCs.

- *Data source:* `experiments/representation_analysis/dimensionality/` — **awaiting updated results**

**Panel C — Activation sparsity.**
Sparsity comparison across layers. How this connects to the broader story (e.g., sparse coding in biological vision) depends on what the updated results show.

- *Data source:* `experiments/representation_analysis/` — **awaiting updated results**

**Panel D (TBD) — Additional characterization.**
Possible candidates once updated results are in:
- RSM comparison (similarity matrices side by side, showing fundamentally different geometry)
- Nearest neighbor retrieval (same query image, different neighbors — 1000-way retrieves same class, coarse retrieves perceptually similar but semantically diverse images)
- Intrinsic dimensionality (Two-NN estimator across layers)

### Existing assets (to be updated)
- `experiments/representation_analysis/2pcs_compare/pc_quadrant_pretrained_vs_4way_*.png` — PC space comparisons
- `experiments/representation_analysis/figs/representation_summary.png` — 4-panel representation analysis
- `experiments/representation_analysis/dimensionality/` — eigenspectrum, participation ratio, intrinsic dimension
- `experiments/representation_analysis/figs/rsm_comparison.png` — RSM comparison
- `experiments/representation_analysis/figs/sparsity_histogram.png` — activation sparsity histograms

---

## Figure 3: Neural Alignment Results (NSD + TVSD)

**Narrative role:** Present the core neural data results. Coarse-trained models match or outperform 1000-way models in brain alignment across species (human fMRI, macaque electrophysiology) and cortical regions (early through ventral/IT). Include inline reconstruction controls showing these results are not explained by dimensionality reduction of the fine-grained model.

### Panels

**Panel A — Dataset schematics (small insets or top row).**
Brief visual descriptions of the two neural datasets:
- *NSD:* Human brain with fMRI overlay → 8 subjects, 7T fMRI, ~10K natural scene images, early + ventral visual streams. Small brain diagram highlighting ROIs.
- *TVSD:* Macaque with electrode arrays → 2 monkeys, Utah arrays, V1/V4/IT, ~22K images, spiking activity.

Visually convey: these are *very different* measurement modalities and species, making convergent results especially compelling.

**Panel B — NSD coarseness results.**
Alignment (Spearman rho) vs. number of classes (log-scale x-axis, 2 → 1000). Two sub-panels: early visual stream, ventral visual stream. Three PCA source models (AlexNet, CLIP, ViT) as separate markers, plus 1000-way baseline and untrained baseline.

- *Key pattern — Early visual stream:* Flat or slightly increasing, saturates early (~8–16 classes). Coarse models match or exceed 1000-way.
- *Key pattern — Ventral visual stream:* Gradual increase from 2 → 64 classes, approaching or matching 1000-way. Even here, 32–64 classes capture most of the alignment.
- Error bars: SEM across 8 subjects × 3 seeds.
- *Data source:* `manuscript/figures/coarseness_log.png`

**Panel C — TVSD coarseness results.**
Same format as Panel B. Three sub-panels: V1, V4, IT.

- *Key pattern — V1:* Coarse models outperform 1000-way (parallels NSD early visual stream).
- *Key pattern — V4:* Increasing with granularity.
- *Key pattern — IT:* Gradual increase, coarse models still competitive.
- Error bars: SEM across 2 monkeys × 3 seeds (wider CIs).
- *Data source:* `manuscript/figures/coarseness_log_tvsd.png`

**Panel D — Reconstruction control (NSD).**
Compact sub-panel or inset: PCA reconstruction of 1000-way model (top-k PCs, k = 1–50) vs. best coarse model, for early and ventral visual streams. Shows that the coarse model's alignment cannot be achieved by simply reducing the dimensionality of the 1000-way model.

- *Data source:* `experiments/reconstruction_analysis/figures/reconstruction_nsd.png`

**Panel E (optional) — Reconstruction control (TVSD).**
Same as Panel D but for TVSD regions.

- *Data source:* `experiments/reconstruction_analysis/figures/reconstruction_tvsd.png`

### Existing assets
- `manuscript/figures/coarseness_log.png` — NSD coarseness (all architectures)
- `manuscript/figures/coarseness_log_tvsd.png` — TVSD coarseness
- `experiments/reconstruction_analysis/figures/reconstruction_nsd.png` — NSD reconstruction control
- `experiments/reconstruction_analysis/figures/reconstruction_tvsd.png` — TVSD reconstruction control
- `plotters/nsd/`, `plotters/tvsd/` — plotting scripts

---

## Figure 4: Behavioral Results and Why Coarse Models Win

**Narrative role:** Present the most surprising finding — coarse-trained models *vastly* outperform the 1000-way model on behavioral alignment (THINGS) — and then explain *why*. This figure transitions from result to mechanism.

### Panels

**Panel A — THINGS coarseness results.**
The headline result: alignment vs. number of classes. All coarse models (2–64 classes, AlexNet/CLIP/ViT) cluster around rho ~ 0.47–0.58, while the 1000-way model drops to ~0.40. Even 2-class models substantially outperform 1000-way.

- *Data source:* `manuscript/figures/coarseness_log_things.png`
- *Key message:* Fine-grained supervision actively *hurts* behavioral alignment. The gap is large (~0.1–0.18 in Spearman rho).

**Panel B — Reconstruction control (THINGS).**
PCA reconstruction of 1000-way model vs. best coarse model. The gap here is the most dramatic of any benchmark: best coarse model (rho ~0.58) far exceeds the 1000-way top-k PCs curve (plateaus ~0.39). This 50% gap cannot be closed by dimensionality reduction.

- *Data source:* `experiments/reconstruction_analysis/figures/reconstruction_things-behavior.png`

**Panel C — Per-concept scatter: which concepts drive the advantage?**
Scatter plot of per-concept RSA: CLIP 4-class (y-axis) vs. 1000-way (x-axis), identity line, color-coded by THINGS semantic category, with marginal histogram of differences.

- *Key pattern:* ~70% of concepts above the diagonal (4-class wins). Plants, animals, clothing accessories strongly favor coarse. Body parts, drinks favor 1000-way.
- *Data source:* `experiments/things_visualizations/figures/per_row_scatter_categories.png`
- *Key message:* The advantage is broad and systematic, not driven by outliers.

**Panel D — Category-annotated RDMs.**
Four-panel: Behavioral | CLIP 4-class | 1000-way | Difference. Concepts sorted by 27 THINGS categories with boundary lines.

- *Key visual:* The 4-class RDM captures the broad block structure of human similarity (animals grouped, food grouped) better than the 1000-way model, which imposes finer distinctions that don't match human judgments.
- *Data source:* `experiments/things_visualizations/figures/rdm_categorized.png`

**Panel E — Semantic dimension profiling.**
Horizontal bar chart: correlation of per-concept advantage (4-class minus 1000-way) with each of the 66 THINGS behavioral dimensions. "Animal-related," "plant-related," "house/furnishing-related" show strong 4-class advantage. "Body-part-related," "fluffy/soft" show 1000-way advantage.

- *Data source:* `experiments/things_visualizations/figures/dimension_profiling.png`
- *Key message:* The coarse model better captures *high-variance* perceptual dimensions that structure human similarity (animacy, naturalness), while 1000-way over-specializes for class identity at the expense of these broader dimensions.

**Panel F (optional) — Image collages.**
Representative images for top concepts where 4-class wins vs. where 1000-way wins. Immediate visual intuition.

- *Data source:* `experiments/things_visualizations/figures/collage_clip4_wins.png`, `collage_1k_wins.png`

### Existing assets
- `manuscript/figures/coarseness_log_things.png` — THINGS coarseness
- `experiments/reconstruction_analysis/figures/reconstruction_things-behavior.png` — reconstruction control
- `experiments/things_visualizations/figures/per_row_scatter_categories.png` — per-concept scatter
- `experiments/things_visualizations/figures/rdm_categorized.png` — categorized RDMs
- `experiments/things_visualizations/figures/dimension_profiling.png` — dimension profiling
- `experiments/things_visualizations/figures/collage_*.png` — image collages

---

## Supplementary Figures (Candidates)

1. **DINOv2 PCA source model results** — all main figure analyses repeated with DINOv2 labels
2. **NSD-Synthetic (OOD) results** — coarse models maintain alignment on synthetic stimuli
3. **Per-subject NSD breakdowns** — dot plots showing consistency across all 8 NSD subjects
4. **Encoding score results** — voxelwise ridge regression on NSD and TVSD
5. **Fine-grained ROI analysis** — V1, V2, V3, hV4, FFA, PPA individually (NSD)
6. **Stimulus robustness** — RSA stability under stimulus subsampling (`experiments/stimulus_robustness/`)
7. **Curriculum learning** — 64-way → 1000-way fine-tuning results (`experiments/coarse_grain_benefits/`)
8. **Nearest neighbor analysis** — image retrieval comparison across models
9. **Per-dimension alignment** — grouped bar charts comparing per-THINGS-dimension RSA for coarse vs. 1000-way

---

## Design Notes

- **Consistent color scheme:** Same palette for PCA source models (AlexNet, CLIP, ViT) and granularity levels across all figures.
- **Log-scale x-axis** for all coarseness plots (Figures 3B–C, 4A).
- **Schematics should be simple** — the reader should grasp the method in 10 seconds (Figure 1).
- **Reconstruction controls inline** — placed as sub-panels within Figures 3 and 4, adjacent to the results they validate. Not in Figure 2 (datasets haven't been introduced yet).
- **Figure 4 is the climax** — give it the most space. The behavioral result is the most surprising and needs the most explanation.
- **No NSD-Synthetic in main figures** — supplementary only.
- **No corruption robustness** — results don't hold up.
