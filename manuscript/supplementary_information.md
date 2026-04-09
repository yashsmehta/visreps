# Supplementary Information

**Carving Nature at Its Joints: A Coarse Feedback Signal for Learning Human-Aligned Visual Representations**

Supplementary Information contains extended discussion of controls, robustness checks, and methodological considerations that accompany the main text. All figures referenced by number live in the Extended Data section (`manuscript/figures/extended_data/`); this document contains no figures itself. Detailed methods, datasets, and statistical procedures are described separately in `methods.md`.

---

## Supplementary Note 1. The coarseness–alignment relationship is robust across PCA source models

*Does the alignment advantage of coarse training depend on which pretrained model is used to define the label boundaries?*

Our coarse-graining procedure extracts principal components from a pretrained model's feature space and uses median splits along these axes to generate labels. A critical question is whether the resulting alignment patterns are contingent on the specific source model — its architecture, training objective, or representational geometry. In the main text, we present three label sources (AlexNet, CLIP, and raw pixels) chosen to span a broad range of representational quality. Here, we evaluate two additional source models: a supervised ViT-L/16 (Dosovitskiy et al., 2021) and DINOv3, a self-supervised vision transformer (Oquab et al., 2024). These four models differ substantially — from a CNN trained on ImageNet classification to a transformer trained via self-supervised distillation — yet all are used identically: only their feature-space PCA defines the coarse labels; the downstream networks are always trained from scratch.

The qualitative pattern is identical across all four sources (Extended Data Fig. 1): early visual regions show flat profiles regardless of granularity, higher visual regions exhibit gradual increases, and coarse models match or exceed the 1,000-class baseline for behavioural alignment. This consistency — despite the fact that AlexNet's PC1 captures low-level visual statistics while CLIP's PC1 captures high-level semantics (Extended Data Fig. 5) — demonstrates that what matters is the *granularity* of the partitioning, not the specific axes along which images are divided.

---

## Supplementary Note 2. The coarseness advantage replicates with WordNet-derived labels

*Is the coarseness–alignment relationship specific to PCA-based label generation, or does it reflect a more general property of coarse supervision?*

All label sources in the main text and Supplementary Note 1 share a common construction: principal component analysis on learned visual features. To test whether the alignment patterns we observe are tied to this specific procedure, we generated an entirely independent set of coarse labels using the WordNet noun hierarchy (Miller, 1995). Starting from the 1,000 ImageNet synsets, we computed pairwise Wu–Palmer semantic similarity scores (Wu & Palmer, 1994) and applied hierarchical agglomerative clustering to produce partitions at multiple granularity levels (2, 3, 4, 10, 20, and 57 classes). This procedure uses no learned visual representations whatsoever — it relies solely on the lexical-semantic distances between English nouns.

Models trained on WordNet-derived labels reproduce the same qualitative pattern (Extended Data Fig. 2): flat alignment in early visual cortex, increasing alignment in higher cortex, and behavioural alignment that matches or exceeds the 1,000-class baseline. The convergence across two fundamentally different label generation procedures — one grounded in visual feature statistics, the other in linguistic taxonomy — provides strong evidence that the coarseness–alignment relationship reflects a genuine property of how coarse category boundaries shape learned representations, rather than an artifact of any particular method for constructing those boundaries.

---

## Supplementary Note 3. Encoding-score analysis replicates the RSA-based neural alignment results

*Is the coarseness–alignment pattern in neural cortex specific to representational similarity analysis, or does it also appear under an encoding-model evaluation?*

Main Figure 3 quantifies neural alignment with representational similarity analysis (RSA), which compares model and brain representational geometries via Spearman correlations between pairwise-distance matrices. RSA is sensitive to the *structure* of a representation but is agnostic to any particular linear mapping between model units and voxels. A complementary evaluation — encoding models — instead fits voxelwise ridge regressions from model activations to measured responses and reports the held-out Pearson correlation between predicted and observed activity. These two metrics emphasise different aspects of alignment, and a finding that depends on only one of them would be difficult to interpret.

We therefore recomputed all neural alignment scores using encoding models (voxelwise `RidgeCV` with SRP-projected activations; see Methods). The qualitative pattern is identical to the RSA results (Extended Data Fig. 3): in both TVSD and NSD, early visual regions show flat coarseness profiles regardless of label granularity, while higher visual regions exhibit the same gradual increase that culminates near or above the 1,000-class baseline. The agreement between RSA and encoding-score evaluations indicates that the coarseness–alignment relationship reflects a genuine property of the learned representations rather than an artifact of any single alignment metric.

---

## Supplementary Note 4. All models successfully learn their classification tasks, but accuracy does not predict alignment

*Do coarsely trained models actually learn, and could their alignment advantage simply be a byproduct of higher classification accuracy on easier tasks?*

Varying granularity from 2 to 1,000 classes changes not only the supervisory signal but also the difficulty of the classification task. Two opposite concerns arise. First, if coarsely trained models failed to learn — producing near-random features — any alignment with brain data would be uninterpretable. Second, if coarse models' alignment advantage were simply a byproduct of higher classification accuracy on easier tasks, the results would have a trivial explanation.

Neither concern is borne out (Extended Data Fig. 4). All models successfully converge, with final test accuracy monotonically decreasing as the number of classes increases (from ~96% at 2-way to ~30% at 1,000-way for AlexNet-derived labels). More informatively, there are large accuracy differences across PCA source models at matched granularity: AlexNet-derived labels produce the easiest tasks (~74% at 64-way), while DINO-derived labels produce the hardest (~40% at 64-way). Despite these substantial accuracy differences, all four label sources produce comparable brain and behavioural alignment (Extended Data Fig. 1). This double dissociation — accuracy varies with label source while alignment does not — rules out classification accuracy as the driver of alignment quality.

---

## Supplementary Note 5. The PCA axes used for coarse-graining capture interpretable visual–semantic structure

*What do the coarse categories actually look like? Are the PCA-derived partitions semantically meaningful or arbitrary?*

Supplementary Notes 1 and 2 establish that the coarseness–alignment relationship holds across multiple label sources. A natural follow-up question is what these label axes actually capture. If the resulting categories were arbitrary or degenerate — for instance, if they simply split images by file size or aspect ratio — the alignment results would be difficult to interpret. To provide intuition, we visualise the images at the extremes ("poles") of each PC axis for two representative source models: AlexNet and CLIP.

For AlexNet, PC1 primarily separates achromatic, manufactured objects from colourful natural scenes — a largely low-level visual distinction that nonetheless tracks a meaningful perceptual boundary (Extended Data Fig. 5). For CLIP, the axes are more semantically structured: PC1 cleanly separates animals and natural scenes from artifacts and text. Later PCs (2–6) capture progressively finer distinctions in both models, and the variance explained decreases accordingly. Crucially, despite these substantial differences in what the axes encode, models trained on labels from both sources achieve comparable neural and behavioural alignment (Extended Data Fig. 1). This reinforces the conclusion from Supplementary Notes 1 and 2: the *scale* of the partitioning — how many categories it produces — matters more than the particular semantic content of the boundaries.

---

## Supplementary Note 6. Alignment is not an artifact of representational dimensionality

*Could coarse training produce low-dimensional representations that align with low-dimensional structure in neural or behavioural data simply by dimensionality matching?*

A potential confound is that coarse-trained models learn representations that are effectively low-dimensional — captured by a handful of principal components — and that these happen to match low-dimensional structure in neural or behavioural data. Under this scenario, the alignment advantage would reflect a dimensionality-matching artifact rather than meaningful representational similarity. To test this, we projected each model's best-layer activations onto the top-*k* principal components (for *k* = 1, 2, …, 50) and recomputed alignment at each dimensionality.

Alignment for both the best coarse model and the 1,000-class baseline plateaus rapidly, typically by *k* ≈ 10–20 PCs (Extended Data Fig. 6). Crucially, the coarse model's advantage over the 1,000-class baseline is maintained across the full range of retained dimensions. The structural correspondence is not confined to the first few PCs; it persists even as the representational space expands to its full rank. This rules out dimensionality matching as an explanation and confirms that the alignment differences reflect genuine representational structure.

---

## Supplementary Note 7. Alignment results are reproducible across random training seeds

*Are the reported alignment patterns stable across independent training runs, or could they be driven by fortuitous random initialisation?*

Every condition in this study is trained with three independent random seeds, controlling for stochasticity in weight initialisation, data shuffling, and dropout. To verify that our conclusions are not sensitive to particular seed choices, we visualise all individual seed scores alongside the condition means.

Across all benchmarks and granularity levels, seed-to-seed variability is consistently small — substantially smaller than the within-seed bootstrap confidence intervals and negligible relative to between-condition effects (Extended Data Fig. 7). The three seed scores cluster tightly around their respective means for both coarse-grained conditions and the 1,000-class baseline. This confirms that training stochasticity does not meaningfully influence the alignment results and that the patterns reported in the main text are reproducible.
