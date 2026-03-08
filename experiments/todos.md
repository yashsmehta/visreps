# TODO List

## 1. Per-Row RDM Correlations (CLIP 4-class vs 1K vs Behavioral)

For each THINGS concept (row in the RDM), correlate the CLIP 4-class model's RDM row against the behavioral RDM row, and separately the 1K model's RDM row against the behavioral RDM row. Take the difference per concept to produce a distribution showing where 4-class wins vs where 1K wins. This directly tests which concepts drive the aggregate RSA advantage.

**Implementation:** Load concept-averaged activations from `experiments/things_visualizations/data/things_viz_data.npz` (`clip4_acts` for the 4-class model, `thousand_acts` for 1K, `embeddings` for behavioral). Build full RDMs using `visreps.analysis.rsa.compute_rdm()`. For each row `i`, extract row vectors from all three RDMs, compute `spearmanr(model_row_i, behav_row_i)` for both models. Store per-concept differences in a DataFrame. Use all ~1854 concepts (no subsampling).

**Enhancements:**
- **Scatter plot:** Plot `corr(4-class row, behav)` vs `corr(1K row, behav)` with identity line; points above diagonal = 4-class wins. Add marginal histogram of differences alongside.
- **Semantic category enrichment:** Group concepts by THINGS superordinate categories and test whether the 4-class advantage concentrates in specific semantic domains (grouped violin/strip plot).
- **Statistical testing:** Wilcoxon signed-rank test on paired differences; FDR correction for individual concept significance.
- **Noise ceiling:** Split-half reliability of behavioral RDM rows (even vs odd images) to bound max achievable per-concept correlation.
- **Annotate extremes:** Label top-10 / bottom-10 concepts on the scatter plot for immediate interpretability.
- **Multi-seed check:** If feasible, verify per-concept pattern is consistent across seeds 1–3 (rank-correlate concept-level differences).

## 2. Characterizing Discrepant Concepts (Image Collages, Dendrograms, Dimension Profiling)

Three complementary views of the per-concept differences from TODO 1, moving from visual to structural to mechanistic.

**A. Image collages.** For the top-N concepts where 4-class outperforms 1K and bottom-N where 1K wins, display grids of representative images (`rep_image_paths` from `things_viz_data.npz`). Immediate visual intuition for what each model handles better.

**B. Dendrograms.** For each tail group (~20 concepts), build 3 mini-RDMs (behavioral, 4-class, 1K), cluster on behavioral ordering, and plot side-by-side dendrograms with concept labels. Use `scipy.cluster.hierarchy.linkage` + `dendrogram` (already used in `plot_rdms.py`). Shows whether the 4-class model preserves the behavioral clustering structure among those concepts better than 1K does.

**C. Semantic dimension profiling.** Correlate the per-concept advantage score (`corr_4class - corr_1K`) with each of the 66 THINGS behavioral dimensions across all ~1854 concepts (no binning). Each dimension has a semantic label (`dimension_labels` from `things_viz_data.npz`). A positive correlation means concepts scoring high on that dimension tend to be better captured by the 4-class model. Plot as a horizontal bar chart of 66 Spearman correlations, sorted by magnitude, with significance markers (FDR-corrected).

## 3. Category-Annotated RDM Visualization with Difference RDM

Replace the current 3-panel heatmap with a richer version: 4 panels (Behavioral | CLIP 4-class | 1K | Difference), concepts sorted by predefined THINGS categories with labeled boundary lines.

**Category annotations:** Use the official 27 THINGS categories (crowdsourced category membership from the THINGS OSF repository: https://osf.io/jum2f/). Sort concepts by category, draw boundary lines between categories on each RDM panel, and label blocks along the axes ("Animals", "Food", "Vehicles", etc.). This makes the RDM immediately interpretable — you can see which semantic blocks are well-structured vs scrambled in each model.

**Difference RDM:** Add a 4th panel showing `rank(4-class RDM) - rank(1K RDM)`. Positive cells = 4-class sees those concepts as more dissimilar than 1K; negative = more similar. With category annotations overlaid, this directly shows which semantic blocks drive the model differences.

**Implementation:** Build on `experiments/things_visualizations/plot_rdms.py` (already has hierarchical clustering, rank-transformation, 3-panel layout). Download THINGS 27-category membership file from OSF. Sort concepts by category instead of hierarchical clustering. Use `ax.axhline`/`ax.axvline` for boundary lines, place category labels at block midpoints. Update panels to use `clip4_acts` (CLIP 4-class) instead of `twoclass_acts`.

## 4. Category-Colored PC Plots for THINGS (4-way Labels)

Project THINGS concept activations onto PC1/PC2 of the 4-class model, color-code each point by its 4-way PCA label (from median split on PCs). This visualizes whether the coarse category structure organizes THINGS concepts into meaningful behavioral clusters (e.g., animate vs inanimate, natural vs man-made).

**Implementation:** Reuse `experiments/representation_analysis/2pcs_compare/run_analysis.py` which already has `compute_pca(features, n_pcs=2)` and `assign_quadrants(pc1, pc2)` via median splits. Load activations from `things_viz_data.npz`, apply PCA, assign quadrants, then scatter plot with 4 colors. Can overlay concept names for key clusters or use `AnnotationBbox` thumbnails as in `plot_embedding_space.py`.

## 5. Dimensional Alignment Between Model and Behavioral Embeddings

Extend the existing per-dimension analysis to test whether the behavioral dimensions (of the 66 THINGS dimensions) that explain more variance in human behavior are specifically the ones where the 4-class model shows improved alignment over 1K. Plot variance-explained vs alignment-improvement to show the correlation.

**Implementation:** `experiments/things_visualizations/plot_dimension_alignment.py` already computes per-dimension Spearman correlations via `compute_per_dimension_rsa()`. Extend this: compute variance explained per dimension (just `np.var(embeddings[:, d])` for each of the 66 dims), compute alignment difference `scores_two[d] - scores_thou[d]`, then scatter plot with `spearmanr` correlation between variance and difference. The existing grouped bar and difference plots serve as the baseline visualization.

## 6. Updated Coarseness Plot with Log-Scale X-Axis

Replace the current bar chart (discrete bars for 2, 4, 8, ..., 64, 1000 categories) with a continuous log-scale x-axis showing all models overlaid. This better communicates the category-count gradient and makes the saturation/plateau effect visually obvious. Optionally overlay multiple architectures (AlexNet, DINOv2, ViT, CLIP) with different markers.

**Implementation:** Data querying uses `plotters/plotter_utils.py`'s `query_best_scores()` for each `(cfg_id, pca_labels_folder)` combo. The x-axis values are `[2, 4, 8, 16, 32, 64, 1000]` on a `log2` scale. Plot as connected lines with markers per architecture, one panel per region/dataset. The existing `plotters/plot_helpers.py` provides `COARSE_CFGS`, color palettes, and spine styling. This is a new plotting script but reuses all the data-access infrastructure.

## 7. Corruption Robustness on THINGS Images (RSA Under Blur/Adversarial)

Apply image corruptions (blur, noise, etc.) to THINGS stimuli and re-compute RSA for both 4-class and 1K models. Hypothesis: the 4-class model maintains behavioral alignment under corruption because it relies on coarse perceptual features rather than fine-grained details that corruptions destroy.

**Implementation:** The corruption infrastructure exists in `experiments/coarse_grain_benefits/imagenet_c_robustness.py` (uses the `imagecorruptions` library with 15 corruption types x 5 severities). Adapt this to THINGS images: load THINGS images via `visreps.dataloaders.neural.load_things_data()`, apply corruptions, extract features using `extract_data.py`'s pipeline (`make_extractor()`, `extract_all_images()`, `concept_average()`), build corrupted RDMs via `compute_rdm()`, and compare to the *original* behavioral RDM with `compute_rdm_correlation()`. Plot RSA vs corruption severity for both models.
