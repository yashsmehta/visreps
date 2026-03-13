# Supplementary Figure Descriptions

Detailed **message** (what we want the reader to take away) and **result** (what the data actually shows) for each supplementary figure. For scripts, layouts, and run commands see `README.md`.

---

## S1. Training Convergence and Classification Accuracy

**Message:** All models successfully learn their respective classification tasks — any differences in brain-model alignment are not attributable to training failure or underfitting. The coarse models are not "broken"; they simply solve an easier task.

**Result:** Test accuracy decreases monotonically from ~96% (2-class) → ~91% (4-class) → ~88% (8-class) → ~85% (16-class) → ~80% (32-class) → ~74% (64-class) → ~74% (1000-way). The 64-class and 1000-way models converge to similar accuracy (~74%), suggesting that 64 PCA-derived classes approach the effective difficulty of full ImageNet classification for this architecture. Error bars across 3 seeds are smaller than markers — high reproducibility. AlexNet-PCA labels shown (representative across PCA sources).

---

## S2. Summary Bar Comparison

**Message:** A concise cross-dataset overview showing that pretrained coarse models match or exceed 1000-way models, while untrained (scratch) models consistently underperform.

**Result:** Bar plots comparing pretrained coarse vs. scratch models across all datasets and regions. Pretrained coarse models achieve comparable or higher alignment than 1000-way across NSD, TVSD, and THINGS, while scratch baselines fall well below both conditions.

---

## S3. Full Per-Layer Profiles (All 7 Granularity Levels)

**Message:** The complete per-layer profiles reveal that all granularity levels converge at the best-aligned layer for each region, and that the layer preference shifts systematically from early (conv1–conv3) for V1 to late (fc1–fc2) for IT/ventral/THINGS.

**Result:** 2×3 grid, all 7 levels (2, 4, 8, 16, 32, 64, 1000) plus untrained. The 7-line profiles are densely overlapping but reveal: (1) TVSD V1 and NSD Early — all lines cluster together at conv2–conv4, with minimal separation between granularity levels. (2) TVSD IT and NSD Ventral — 1000-way (orange) achieves highest scores at fc1–fc2, while coarse models show competitive or higher scores at intermediate layers (conv4–conv5). (3) THINGS — most dramatic separation: coarse models (blue gradient) dominate at fc1–fc2 with scores of ~0.55–0.60, while 1000-way (orange) peaks lower at ~0.38. The blue gradient ordering (lighter = coarser = higher) is clearest in THINGS fc2.

---

## S4. Neural Reconstruction

**Message:** Alignment depends on how many principal components are retained when reconstructing labels, revealing the effective dimensionality needed for brain-model alignment in each region.

**Result:** Plots of alignment score vs. number of PCs retained for TVSD and NSD, showing how neural alignment changes as more principal components are included in label reconstruction.

---

## S5. THINGS Per-Architecture Breakdown

**Message:** The coarse model behavioral advantage is robust across PCA source models — it is not driven by a single architecture. Even Pixels-PCA, the weakest source, shows meaningful structure.

**Result:** 1×4 grid, one panel per PCA source. (A) AlexNet — all coarse levels at 113–122% of 1000-way, with a slight peak at 32–64 classes. (B) CLIP — strongest advantage: all coarse levels at ~140–145%, essentially flat. Even 2-class CLIP exceeds 1000-way by ~40%. (C) ViT — similar to CLIP: ~135–148%, with a slight uptick at 64 classes. (D) Pixels — weakest: 2–8 class models at ~25–30%, rising to ~55–60% at 32–64 classes. Pixels never reaches 100% (the 1000-way baseline), confirming that raw pixel statistics produce less brain-aligned categories than pretrained feature statistics. However, the ramp from 25% to 60% shows that even pixel-based coarsening captures some meaningful structure.

---

## S6. Fine-Grained ROI Decomposition (NSD)

**Message:** The coarseness effect shows a clear gradient along the visual hierarchy at fine anatomical resolution: early retinotopic areas (V1–V3) saturate at low granularity, while category-selective regions (FFA, PPA) require more classes.

**Result:** 2×3 grid, 6 individual ROIs, all 4 architectures. (A) V1 — flat at ~90–105%, all coarse levels match 1000-way. (B) V2 — similar, ~85–100%. (C) V3 — slightly more ramp, ~80–100%. (D) hV4 — intermediate, ~75–105% with clear convergence by 16-class. (E) FFA — steepest ramp: starts at ~45–50% for 2-class, reaches ~95–105% by 64-class. This makes sense — face-selective regions need sufficient granularity to distinguish face-relevant categories. (F) PPA — starts at ~65–70% for 2-class, reaches ~95–100% by 32-class. Place-selective but less granularity-dependent than FFA. Pixels (brown) consistently underperforms other architectures, especially in FFA/PPA.

---

## S7. NSD-Synthetic: Out-of-Distribution Stimuli

**Message:** The coarse model alignment advantage is not an artifact of ImageNet-like stimulus statistics. Even on synthetic stimuli (noise patterns, gratings, Mooney images) that are completely unlike the training distribution, coarse models maintain comparable neural alignment.

**Result:** (A) Early visual stream — high variability (large error bars) with all conditions in the 60–120% range; untrained baseline is 325% of 1000-way (annotated off-scale), reflecting that random networks achieve disproportionately high RSA on synthetic stimuli in early visual cortex. (B) Ventral visual stream — clearer pattern: AlexNet-PCA (teal) and CLIP-PCA (purple) both ramp from ~70% at 2-class to ~90–105% at 32–64 classes, closely mirroring the natural-image NSD results. CLIP labels slightly outperform AlexNet at higher granularity. The untrained baseline is well separated at ~58%.

---

## S8. Stimulus Robustness (Subsampling Analysis)

**Message:** The RSA alignment estimates and the coarse ≥ fine pattern are robust to stimulus set size — not an artifact of which specific stimuli happen to be in the test set.

**Result:** Line plots showing RSA as a function of stimulus subsample fraction (10%–100%), with 50 repetitions per fraction. Mean and 95% CI bands are shown. For all regions, RSA estimates stabilize by ~30–50% subsampling. The ordering between coarse and 1000-way conditions is preserved at all subsampling fractions, confirming that the coarseness effect is not driven by a few influential stimuli.

---

## S9. Score Distributions (Violin/Strip Plots)

**Message:** The main-figure point estimates and error bars faithfully represent the underlying data — the effects are not artifacts of averaging over heterogeneous subjects or seeds.

**Result:** 2×3 grid showing full score distributions. (A) TVSD V1 — violin plots with 6 points each (2 monkeys × 3 seeds); distributions broadly overlap across granularity levels, consistent with the flat coarseness curve. (B) TVSD V4 — slight upward shift with increasing classes, 1000-way distribution overlaps with 32–64 class distributions. (C) TVSD IT — clearest monotonic shift: 2-class distributions centered at ~0.12, increasing to ~0.17 at 64-class and 1000-way. (D) NSD Early — violins with 24 points (8 subjects × 3 seeds); wide spread but all conditions overlap. (E) NSD Ventral — gradual rightward shift in distribution centers, replicating the coarseness ramp. (F) THINGS — 3 points per condition (3 seeds); coarse conditions tightly clustered at ~0.53–0.57, well above 1000-way at ~0.38. Untrained is clearly separated in all panels.

---

## S10. DINOv3 as an Additional PCA Source Model

**Message:** The coarseness–alignment relationship is not contingent on the representational geometry of any single PCA source model. Even a self-supervised vision transformer (DINOv3), which was trained without category labels, produces coarse labels that yield the same qualitative pattern.

**Result:** Across all 6 dataset–region combinations, DINOv3-derived labels (cyan pentagons) replicate the main findings: (A) NSD early visual stream — flat curve at ~85–100% of 1000-way; (B) NSD ventral — gradual ramp from ~55% to ~95%; (C) TVSD V1 — noisy but centered around 100% with large error bars; (D) TVSD V4 — ramp from ~75% to ~95%; (E) TVSD IT — ramp from ~55% to ~100%; (F) THINGS behavioral — all coarse levels at ~100–120%, exceeding 1000-way. Error bars are larger than main-figure architectures (fewer runs available), but the qualitative pattern is consistent.

---

## S11. Encoding Score Coarseness Curves (NSD + TVSD)

**Message:** A fundamentally different alignment metric — encoding score (linear ridge regression, Pearson r) — corroborates the RSA findings. Coarse models contain linearly recoverable neural information at levels comparable to 1000-way training.

**Result:** 2×3 grid (5 panels), AlexNet + CLIP PCA labels. (A) NSD early visual stream — essentially flat at ~97–102%, even 2-class models match 1000-way. (B) NSD ventral visual stream — gradual ramp from ~85% (2-class) to ~100% (64-class). (C) TVSD V1 — flat at ~99–101%. (D) TVSD V4 — ramp from ~92% to ~100%. (E) TVSD IT — ramp from ~91% to ~100%. The qualitative pattern mirrors RSA exactly: early regions saturate at low granularity, higher regions show a gradual increase. AlexNet and CLIP labels produce nearly identical curves. The untrained baseline is well separated (dashed gray line at ~84% for NSD early, ~55% for NSD ventral, ~95% for TVSD V1, ~72% for TVSD V4/IT).

---

## S12. WordNet Hierarchy

**Message:** WordNet-derived coarse labels — constructed from the existing taxonomic hierarchy rather than learned feature statistics — serve as an alternative label source and produce comparable alignment patterns, confirming that the coarseness effect is not specific to PCA-based label generation.

**Result:** Alignment curves using WordNet-derived coarse labels as an alternative to PCA-based labels, showing that hierarchical linguistic/taxonomic groupings of ImageNet classes yield similar coarseness–alignment relationships across datasets.

---

## S13. Internal Representation Analysis Across Granularity

**Message:** Coarse training produces representations that concentrate variance in fewer dimensions with higher sparsity — a fundamentally different representational geometry, not just a lower-dimensional version of the same geometry.

**Result:** 2×2 grid, all 7 granularity levels (blue gradient + orange). (A) FC1 eigenspectrum — all coarse models (2–64, blue lines) have steeper eigenvalue decay than 1000-way (orange). The 2-way model concentrates ~50% of variance in the first PC. 1000-way has a gradual, near-uniform decay across ~100 components. (B) Participation ratio across layers — all models are similar through conv1–conv4. Divergence begins at conv5 and is maximal at FC1–FC2: coarse models drop to PR ~10–30, while 1000-way maintains PR ~100+. (C) TwoNN intrinsic dimension — similar pattern: convergence at early layers, divergence at FC layers. Coarse → ~15–20, 1000-way → ~40 at FC2. (D) Sparsity — coarse models reach ~0.85 Hoyer sparsity at FC2, vs. ~0.60 for 1000-way. Early layers show no difference. Untrained (gray dashed) has the lowest PR and highest sparsity of all — suggesting that training generally increases dimensionality, but coarse training increases it less.

---

## S14. THINGS Behavioral Dimension Profiling

**Message:** The coarse model advantage is driven by high-level categorical dimensions (animal, plant, natural) while the 1000-way advantage is driven by lower-level material/functional properties (home furnishing, metallic, household). This reveals *what* the coarse model captures better.

**Result:** Horizontal bar chart of top 25 dimensions by |ρ|, all 66/66 significant after FDR correction. Green bars (positive ρ) = high loading favors coarse model: **Animal** (ρ ≈ +0.25), **Plant** (ρ ≈ +0.24), **Bug/non-mammalian** (ρ ≈ +0.19), White, Grid/grating. These are high-variance categorical dimensions that structure similarity at the superordinate level. Red bars (negative ρ) = high loading favors 1000-way: **Home/furnishing** (ρ ≈ −0.30), **Metallic/artificial** (ρ ≈ −0.25), **Household** (ρ ≈ −0.22), Fluid/drink, Bathroom. These are material/functional properties that require within-category discrimination — exactly what 1000-way training emphasizes.

---

## S15. Image Collages (Coarse-Wins vs. 1000-Way-Wins Concepts)

**Message:** The concepts where coarse models win are visually and categorically coherent (animals, plants, vehicles), while 1000-way-winning concepts tend to be objects defined by fine-grained material or functional properties.

**Result:** Two composite panels. (A) Coarse-wins concepts — dominated by animals (birds, insects, mammals), plants, and large outdoor objects. These are categories whose similarity structure is well-captured by broad categorical boundaries. (B) 1000-way-wins concepts — include body parts, kitchen utensils, tools, and household items. These require fine-grained discrimination between visually similar but functionally distinct objects.

---

## S16. PC Axis Interpretation (Pole Images)

**Message:** The PCA-based coarse labels are semantically meaningful — the principal components separate images along interpretable visual/semantic axes, not arbitrary statistical dimensions.

**Result:** Top PCs for AlexNet and CLIP, showing most-activating (positive pole) and least-activating (negative pole) ImageNet images. PC1 for both models separates natural scenes (animals, landscapes) from man-made objects (electronics, vehicles). PC2 separates indoor/structured from outdoor/organic. Higher PCs (3–6) capture progressively finer axes: texture (smooth vs. rough), color temperature (warm vs. cool), and category-specific distinctions. CLIP PCs tend to be more semantically crisp than AlexNet PCs.

---

## S17. Levels Evaluation (Hierarchical Similarity Benchmark)

**Message:** Coarse models improve on human similarity judgments specifically at the between-category level, where broad categorical structure determines the correct response — consistent with the THINGS findings.

**Result:** 3×3 grid (3 metrics × 3 triplet types). Metrics: odd-one-out accuracy, uncertainty alignment (Spearman r with human RT), triplet RSA. Triplet types: within-class, class-boundary, between-class. Key patterns: (1) **Between-class** — coarse models (AlexNet, CLIP, DINO, ViT) all exceed the 1000-way dashed baseline by 16–64 classes. AlexNet and CLIP show the strongest gains. (2) **Class-boundary** — similar but attenuated improvement. (3) **Within-class** — coarse models start below 1000-way at 2 classes but converge by 32–64 classes. This three-way pattern confirms that coarse training specifically enhances the broad categorical structure relevant to between-class similarity, while preserving sufficient within-category information at moderate granularity.
