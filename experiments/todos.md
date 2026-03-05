# TODO List

## 1. Per-Row RDM Correlations (4-class vs 1K vs Behavioral)

For each THINGS concept (row in the RDM), correlate the 4-class model's RDM row against the behavioral RDM row, and separately the 1K model's RDM row against the behavioral RDM row. Take the difference per concept to produce a distribution showing where 4-class wins vs where 1K wins. This directly tests which images/concepts drive the aggregate RSA advantage.

**Implementation:** Load concept-averaged activations from `experiments/things_visualizations/data/things_viz_data.npz` (already has `twoclass_acts`, `thousand_acts`, `embeddings`). Build full RDMs using `visreps.analysis.rsa.compute_rdm()`. For each row `i`, extract row vectors from all three RDMs, compute `spearmanr(model_row_i, behav_row_i)` for both models. Store per-concept differences in a DataFrame. The `plot_rdms.py` pattern of subsampling + hierarchical clustering can be adapted for visualization.

## 2. Dendrogram Visualizations (Most/Least Discrepant Concepts)

From the per-row correlation distribution (TODO 1), select the top-N concepts where 4-class outperforms 1K (most positive difference) and top-N where 1K outperforms 4-class (most negative). Build small dendrograms for each group showing how those concepts cluster under each model vs behavior. This reveals *what kinds of images* each model handles better.

**Implementation:** Use `scipy.cluster.hierarchy.linkage` + `dendrogram` (already used in `plot_rdms.py` lines 58-60 for hierarchical clustering with `squareform` + `linkage(method="average")`). For each tail group (~20 concepts), build 3 mini-RDMs (behavioral, 4-class, 1K), cluster on behavioral, and plot side-by-side dendrograms with concept labels. Concept names come from `things_viz_data.npz["concept_names"]`.

## 3. More Informative RDM Visualization

Create a richer RDM visualization that goes beyond the current 3-panel heatmap. Options include: annotating cluster boundaries with category labels, adding marginal dendrograms, showing the difference RDM (4-class minus 1K), or using a split-matrix format (behavioral upper triangle, model lower triangle).

**Implementation:** Build on `experiments/things_visualizations/plot_rdms.py` which already does hierarchical clustering (`linkage` + `leaves_list`), rank-transformation, and 3-panel heatmaps. Add a 4th panel for the difference RDM (`rdm_two_r - rdm_thou_r`). For annotated cluster boundaries, use `fcluster()` on the dendrogram linkage to identify cluster breaks, then draw boundary lines on the heatmap with `ax.axhline`/`ax.axvline`. Category labels from `concept_names` can be placed at cluster midpoints along the axes.

## 4. Category-Colored PC Plots for THINGS (4-way Labels)

Project THINGS concept activations onto PC1/PC2 of the 4-class model, color-code each point by its 4-way PCA label (from median split on PCs). This visualizes whether the coarse category structure organizes THINGS concepts into meaningful behavioral clusters (e.g., animate vs inanimate, natural vs man-made).

**Implementation:** Reuse `experiments/representation_analysis/2pcs_compare/run_analysis.py` which already has `compute_pca(features, n_pcs=2)` and `assign_quadrants(pc1, pc2)` via median splits. Load activations from `things_viz_data.npz`, apply PCA, assign quadrants, then scatter plot with 4 colors. Can overlay concept names for key clusters or use `AnnotationBbox` thumbnails as in `plot_embedding_space.py`.

## 5. Body Parts & Faces in PC / UMAP Space (THINGS)

Investigate where body parts and faces fall in PCA and UMAP projections of THINGS concept embeddings for both the 4-class and 1K models. Highlight these categories to see if the 4-class model groups them similarly to human behavioral data (where body parts/faces are a very prominent cluster).

**Implementation:** Use the UMAP pipeline from `experiments/things_visualizations/plot_embedding_space.py` (`run_umap()` with PCA pre-reduction). Identify body-part/face concepts from `concept_names` using keyword matching or the THINGS category metadata. Plot all concepts as grey scatter, overlay body/face concepts in a highlight color. Generate one panel per model (behavioral, 4-class, 1K). The `grid_subsample()` pattern from `plot_embedding_space.py` can place thumbnails of body-part images at their UMAP coordinates.

## 6. Dimensional Alignment Between Model and Behavioral Embeddings

Extend the existing per-dimension analysis to test whether the behavioral dimensions (of the 66 THINGS dimensions) that explain more variance in human behavior are specifically the ones where the 4-class model shows improved alignment over 1K. Plot variance-explained vs alignment-improvement to show the correlation.

**Implementation:** `experiments/things_visualizations/plot_dimension_alignment.py` already computes per-dimension Spearman correlations via `compute_per_dimension_rsa()`. Extend this: compute variance explained per dimension (just `np.var(embeddings[:, d])` for each of the 66 dims), compute alignment difference `scores_two[d] - scores_thou[d]`, then scatter plot with `spearmanr` correlation between variance and difference. The existing grouped bar and difference plots serve as the baseline visualization.

## 7. PCA Reconstruction Plot (RSA vs Number of PCs)

Regenerate the plot showing RSA score as a function of the number of PCs (`k`) used to reconstruct 1K-model representations. This tests whether the coarse model's advantage is simply captured by the top few PCs of the fine-grained model. If the reconstruction curve plateaus below the coarse model's score, it means the coarse model learns something qualitatively different.

**Implementation:** `experiments/reconstruction_analysis/plot.py` already has the full pipeline: `query_reconstruction_curve()`, `aggregate_curve()`, `query_1000way_baseline()`, `query_coarse_baseline()`, `query_untrained_baseline()`, and `plot_panel()`. This is essentially a re-run of the existing plotter, possibly with updated style or additional data points. The data comes from `results.db` (runs with `reconstruct_from_pcs=1`). If new reconstruction evals are needed, use the standard eval pipeline with `reconstruct_from_pcs=true` and `pca_k` values in a grid config.

## 8. Updated Coarseness Plot with Log-Scale X-Axis

Replace the current bar chart (discrete bars for 2, 4, 8, ..., 64, 1000 categories) with a continuous log-scale x-axis showing all models overlaid. This better communicates the category-count gradient and makes the saturation/plateau effect visually obvious. Optionally overlay multiple architectures (AlexNet, DINOv2, ViT, CLIP) with different markers.

**Implementation:** Data querying uses `plotters/plotter_utils.py`'s `query_best_scores()` for each `(cfg_id, pca_labels_folder)` combo. The x-axis values are `[2, 4, 8, 16, 32, 64, 1000]` on a `log2` scale. Plot as connected lines with markers per architecture, one panel per region/dataset. The existing `plotters/plot_helpers.py` provides `COARSE_CFGS`, color palettes, and spine styling. This is a new plotting script but reuses all the data-access infrastructure.

## 9. Corruption Robustness on THINGS Images (RSA Under Blur/Adversarial)

Apply image corruptions (blur, noise, etc.) to THINGS stimuli and re-compute RSA for both 4-class and 1K models. Hypothesis: the 4-class model maintains behavioral alignment under corruption because it relies on coarse perceptual features rather than fine-grained details that corruptions destroy.

**Implementation:** The corruption infrastructure exists in `experiments/coarse_grain_benefits/imagenet_c_robustness.py` (uses the `imagecorruptions` library with 15 corruption types x 5 severities). Adapt this to THINGS images: load THINGS images via `visreps.dataloaders.neural.load_things_data()`, apply corruptions, extract features using `extract_data.py`'s pipeline (`make_extractor()`, `extract_all_images()`, `concept_average()`), build corrupted RDMs via `compute_rdm()`, and compare to the *original* behavioral RDM with `compute_rdm_correlation()`. Plot RSA vs corruption severity for both models.

## 10. THINGS fMRI Analysis (NSD with THINGS Stimuli)

Run the standard RSA evaluation on NSD fMRI data specifically for THINGS stimuli that overlap with NSD, to check whether the behavioral RSA advantage of coarse models also appears in the neural (fMRI) domain for the same stimulus set. This bridges the behavioral and neural results.

**Implementation:** This is a standard eval pipeline run. Use `scripts/runners/eval_runner.py` with a grid config specifying `neural_dataset: "nsd"`, `analysis: "rsa"`, and the appropriate `cfg_id` values. The NSD dataset already contains ~9K stimuli; the overlap with THINGS concepts would need to be identified (match NSD COCO images to THINGS concepts). Alternatively, use the existing NSD results from `results.db` and compare the coarse vs fine-grained patterns to the THINGS behavioral results.
