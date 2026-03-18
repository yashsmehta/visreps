# Supplementary Materials

**Coarse-Grained Visual Supervision Is Sufficient for Brain- and Behavior-Aligned Representations**

---

## Overview

This supplement provides additional analyses supporting the main findings. The supplementary figures are organized into six themes: (1) **training validation** (Figure S1), confirming that all models converge; (2) **extended main results** (Figures S2–S5), providing the cross-dataset summary, per-layer profiles, reconstruction controls, and per-architecture breakdowns omitted from main figures for conciseness; (3) **anatomical detail** (Figure S6), showing coarseness effects at finer ROI resolution; (4) **robustness and generalization** (Figures S7–S9, S17), testing whether results hold under out-of-distribution stimuli, stimulus subsampling, across individual subjects and seeds, and across random training seeds; (5) **alternative labels** (Figures S10–S11), demonstrating that findings replicate with ViT, DINOv3, and WordNet-derived labels; and (6) **representational and perceptual analysis** (Figures S12–S16), characterizing how coarse training reshapes internal representations, which behavioral dimensions drive the alignment advantage, and how the label space is structured.

All alignment scores use Spearman $\rho$ (RSA) unless otherwise noted. Error bars denote 95% bootstrap CIs (NSD, THINGS) or $\pm$1.96 SEM across monkeys $\times$ seeds (TVSD). Code for all figures is available in `manuscript/figures/supplementary/`.

---

## Figure S1. Training convergence and classification accuracy

![Figure S1](figures/supplementary/supp_s1_training_summary.png)

**Figure S1. All models successfully learn their respective classification tasks, with accuracy monotonically decreasing as granularity increases.** Final test accuracy (epoch 20, mean $\pm$ SEM across 3 seeds) is plotted as a function of the number of output classes on a log$_2$ scale. Models trained on AlexNet-PCA labels are shown (results are representative across PCA source models). Two-class models achieve $\sim$96% accuracy, reflecting the relative ease of a binary classification task on ImageNet, while 1000-way models reach $\sim$74% — consistent with AlexNet-class architectures trained for 20 epochs. The monotonic decrease confirms that all coarse models converge to their respective tasks and that any differences in brain-model alignment (Figures 3–4) are not attributable to training failure. Error bars are smaller than markers, indicating high reproducibility across seeds.

---

## Figure S2. Cross-dataset summary bar comparison

![Figure S2](figures/supplementary/supp_s2_summary_bars.png)

**Figure S2. Summary comparison of pretrained and trained-from-scratch models across all five benchmarks.** Five-condition bar plots for TVSD V1 **(A)**, TVSD IT **(B)**, NSD early visual stream **(C)**, NSD ventral visual stream **(D)**, and THINGS behavioral similarity **(E)**. Each panel compares two pretrained models (ViT-Base, CLIP ViT-L/14; translucent bars) against three trained-from-scratch conditions (untrained, 1000-way, and best coarse model; hatched bars). The best coarse model identity is annotated above each bar. Across all benchmarks, the best coarse model matches or exceeds the 1000-way baseline, and in several cases approaches pretrained transformer performance despite being trained from scratch with far fewer category labels. Error bars denote 95% bootstrap CIs (NSD, THINGS) or $\pm$1.96 SEM (TVSD). This figure provides a complementary view to the coarseness curves in Figures 3–4, summarizing the key finding in a single glance: fine-grained supervision is not necessary for competitive brain- and behavior-aligned representations.

---

## Figure S3. Full per-layer profiles for all 7 granularity levels

![Figure S3](figures/supplementary/supp_s3_full_per_layer.png)

**Figure S3. Per-layer RSA profiles showing all seven granularity levels across all datasets and regions.** The complete set of granularity levels: 2, 4, 8, 16, 32, 64-way (blue gradient, light to dark) and 1000-way (orange-red), plus the untrained baseline (gray dashed). Panels cover TVSD V1, TVSD V4, TVSD IT (top row), NSD early visual stream, NSD ventral visual stream, and THINGS (bottom row). The best PCA source architecture is auto-selected per region. Several patterns emerge: (1) in early visual regions (TVSD V1, NSD Early), per-layer profiles are largely overlapping across granularity levels, confirming the flat coarseness curve from a complementary perspective; (2) in higher visual regions (TVSD IT, NSD Ventral), the profiles fan out in later layers (fc1, fc2), with coarser models peaking at intermediate layers and finer models at deeper layers; (3) for THINGS, coarse models consistently achieve higher peak RSA than 1000-way across the full layer hierarchy; (4) the 1000-way model (orange) typically shows a distinctive late-layer peak, reflecting its fine-grained classification objective.

---

## Figure S4. Neural reconstruction analysis (TVSD and NSD)

![Figure S4](figures/supplementary/supp_s4_neural_reconstruction.png)

**Figure S4. Reconstruction control confirms that alignment is not an artifact of dimensionality reduction.** Alignment (Spearman $\rho$) as a function of the number of PCs retained (top-$k$) for the best coarse model (blue) vs. 1000-way (orange) across all neural dataset–region pairs: TVSD V1 **(A)**, TVSD V4 **(B)**, TVSD IT **(C)**, NSD early visual stream **(D)**, and NSD ventral visual stream **(E)**. The untrained baseline is shown as a gray dashed line; shaded bands indicate 95% CIs. For most regions, both the coarse and 1000-way curves plateau by $k \approx 10$–20 PCs, indicating that the representational structure captured by a small number of principal dimensions is sufficient for alignment. In TVSD V1, the coarse 64-way model slightly underperforms 1000-way across all $k$ values, whereas in NSD ventral visual stream, the coarse model (CLIP 16-way) converges to 1000-way by $k = 10$. These reconstruction curves complement Figure 4A (THINGS reconstruction) and demonstrate that the main alignment results are robust to the number of representational dimensions considered.

---

## Figure S5. Per-PCA-source THINGS coarseness functions

![Figure S5](figures/supplementary/supp_s5_things_architectures.png)

**Figure S5. The coarse behavioral alignment advantage holds across all four PCA source architectures, with the exception of raw Pixels at very low granularity.** Individual normalized coarseness curves for THINGS behavioral alignment, showing each PCA source model separately: AlexNet **(A)**, CLIP **(B)**, ViT **(C)**, and Pixels **(D)**. Main Figure 4 overlays all four architectures; this figure isolates each to reveal per-architecture dynamics. AlexNet, CLIP, and ViT labels all produce coarse models that substantially exceed 100% (the 1000-way baseline) at every granularity level, with CLIP and ViT showing the largest advantage ($\sim$140% at peak). The Pixels condition — in which coarse labels are derived from PCA of raw pixel values rather than learned features — shows a qualitatively different pattern: all granularity levels fall well below 100%, with 2–8 class models at $\sim$25–30% and a partial recovery to $\sim$55–60% at 16–64 classes. Even at peak granularity, Pixel-PCA labels do not approach the 1000-way baseline, indicating that pixel-level statistics alone are insufficient to define behaviorally relevant categories.

---

## Figure S6. Fine-grained ROI decomposition (NSD)

![Figure S6](figures/supplementary/supp_s6_finegrained_roi.png)

**Figure S6. Coarseness effects at finer anatomical resolution across six individual ROIs.** Normalized coarseness curves for V1 **(A)**, V2 **(B)**, V3 **(C)**, hV4 **(D)**, FFA **(E)**, and PPA **(F)**, all from NSD. Four PCA source architectures are plotted (AlexNet, CLIP, ViT, Pixels). Main Figure 3 collapses these into two broad streams (early = V1+V2+V3; ventral = hV4+higher areas); this figure reveals the pattern at the level of individual retinotopic and category-selective regions. Early visual areas (V1–V3) show the characteristic flat profile: even 2-class models achieve $\sim$90–100% of 1000-way alignment. Intermediate area hV4 shows a slight increase with granularity but saturates by 16–32 classes. The most pronounced granularity dependence appears in category-selective cortex: FFA shows a steep ramp from $\sim$50% (at 2 classes) to $\sim$100% (at 32–64 classes), and PPA from $\sim$65–70% to $\sim$100%, consistent with these regions' known selectivity for fine-grained object categories (faces, places). Notably, even for FFA and PPA, 32-class models approach the 1000-way ceiling, suggesting that the full 1000-category supervisory signal is not necessary even for aligning with the most category-selective cortical regions.

---

## Figure S7. NSD-Synthetic: coarse model alignment on out-of-distribution stimuli

![Figure S7](figures/supplementary/supp_s7_nsd_synthetic.png)

**Figure S7. Coarse models maintain neural alignment on out-of-distribution synthetic stimuli.** Normalized coarseness curves for NSD-Synthetic (Gifford et al., 2026), a companion dataset to NSD comprising 220 synthetic stimuli (noise patterns, gratings, line drawings, Mooney images) presented to the same 8 subjects under identical scanning conditions. Results are shown for the early visual stream **(A)** and ventral visual stream **(B)**, using AlexNet-PCA (teal circles) and CLIP-PCA (purple squares) labels. In the ventral visual stream, the coarseness–alignment relationship closely mirrors that observed for natural images (Figure 3): low-granularity models approach or exceed the 1000-way baseline. In the early visual stream, all models cluster in the 60–120% range, with substantial variability reflecting the lower signal-to-noise ratio of early visual RSA on synthetic stimuli. Note: the untrained baseline for the early visual stream is 325% of 1000-way (annotated but not shown on the y-axis), reflecting the fact that untrained networks' random representations can achieve higher-than-trained RSA in early visual cortex for synthetic stimuli that lack natural image statistics.

---

## Figure S8. RSA stability under stimulus subsampling

![Figure S8](figures/supplementary/supp_s8_stimulus_robustness.png)

**Figure S8. The coarseness–alignment result is robust to stimulus subsampling.** To test whether results depend on the particular set of test stimuli, we subsample test images at fractions from 10% to 100% and recompute RSA at each fraction, repeating 50 times per fraction to obtain confidence bands. Two representative benchmark–region pairs are shown: NSD ventral visual stream (left) and TVSD IT (right). For each, we compare the best coarse model (blue) against the 1000-way baseline (orange). Mean scores (lines) and 95% CIs (shaded bands) are shown across the 50 subsampling repetitions. In both cases, the coarse model's advantage is stable across the full range of subsample sizes: means fluctuate minimally and the ordering is preserved at every fraction. CIs widen at small fractions — as expected when computing RDMs from fewer stimuli — but the two models' CIs become non-overlapping by $\sim$30% for NSD. This analysis confirms that the main results are not driven by a small subset of influential stimuli.

---

## Figure S9. Score distributions across subjects and seeds

![Figure S9](figures/supplementary/supp_s9_score_distributions.png)

**Figure S9. Full distributions of alignment scores across all subjects and seeds.** Violin plots (with individual data points overlaid) showing the complete spread of RSA scores for each granularity level. Panels show TVSD V1, V4, and IT (top row; 2 monkeys $\times$ 3 seeds = 6 points per condition) and NSD early visual stream, NSD ventral visual stream, and THINGS (bottom row; NSD: 8 subjects $\times$ 3 seeds = 24 points; THINGS: 3 seeds). The best PCA source architecture is auto-selected per region. For NSD, the violins reveal unimodal distributions at each granularity level — no bimodality or heavy tails that might indicate subject-driven artifacts. The progressive increase in both mean and spread for ventral visual stream scores as granularity increases is clearly visible. The untrained baseline (gray) is clearly separated from all trained conditions across all panels.

---

## Figure S10. Additional PCA source models (ViT and DINOv3)

![Figure S10](figures/supplementary/supp_s10_dinov2.png)

**Figure S10. The coarseness–alignment relationship replicates with ViT- and DINOv3-derived labels across all benchmarks.** Normalized coarseness curves (% of 1000-way) for models trained on labels generated from ViT-Base (crimson triangles) and DINOv3 (ViT-L/16, self-supervised; cyan pentagons) are shown for all six dataset–region combinations: NSD early visual stream **(A)**, NSD ventral visual stream **(B)**, TVSD V1 **(C)**, TVSD V4 **(D)**, TVSD IT **(E)**, and THINGS behavioral similarity **(F)**. Both architectures produce the same qualitative pattern seen with AlexNet and CLIP in the main figures: early visual regions saturate at low granularity, higher visual regions show a gradual increase, and THINGS behavioral alignment exceeds 100% at most coarseness levels. ViT labels generally track slightly above DINOv3. This demonstrates that the main findings are not contingent on the specific representational geometry of any single PCA source model, extending robustness to both supervised and self-supervised vision transformers. ViT and DINOv3 labels are omitted from main figures for visual clarity.

---

## Figure S11. WordNet hierarchy as alternative coarse label source

![Figure S11](figures/supplementary/supp_s11_wordnet.png)

**Figure S11. Models trained on WordNet-derived coarse labels show comparable alignment patterns to PCA-based labels.** Coarseness curves for models trained on labels derived from the WordNet taxonomic hierarchy (Wu–Palmer similarity-based clustering) rather than PCA-based partitioning. Results are shown across all neural and behavioral benchmarks. The qualitative pattern — early visual regions saturate at low granularity, higher regions show gradual increases — replicates with this entirely independent label source, further demonstrating that the coarseness–alignment relationship is not an artifact of the PCA-based label generation procedure.

---

## Figure S12. Internal representation geometry across granularity levels

![Figure S12](figures/supplementary/supp_s12_representation_summary.png)

**Figure S12. Coarse training produces lower-dimensional, sparser representations with concentrated variance.** Four representational geometry metrics are shown across all seven granularity levels (2–1000) plus the untrained baseline, using AlexNet-PCA labels. **(A)** FC1 eigenspectrum (log-log): coarser models exhibit steeper eigenvalue decay, concentrating representational variance in fewer dimensions. **(B)** Effective dimensionality (participation ratio) across layers: coarser models show progressively lower effective dimensionality, particularly in later layers (fc1, fc2). **(C)** Two-nearest-neighbor intrinsic dimension: confirms the participation ratio finding using a nonlinear estimator — coarser models occupy lower-dimensional manifolds. **(D)** Hoyer sparsity: later-layer sparsity increases with training but shows limited dependence on granularity, suggesting sparsity is primarily driven by the ReLU nonlinearity rather than label structure. Together, these metrics characterize the representational signature of coarse-grained training: a compressed, low-dimensional geometry in which a small number of principal axes capture the majority of activation variance.

---

## ~~Figure S13.~~ *Moved to main Figure 4C.*

---

## Figure S14. Image collages for coarse-advantaged and 1000-way-advantaged concepts

![Figure S14](figures/supplementary/supp_s14_image_collages.png)

**Figure S14. Visual examples of concepts where coarse and fine-grained models diverge.** Representative THINGS images for the top 20 concepts where the CLIP 4-way model most outperforms 1000-way **(A)** and the top 20 where 1000-way most outperforms CLIP 4-way **(B)**. Numbers below each image indicate the per-concept RSA contribution. Coarse-advantaged concepts are dominated by natural categories — animals, plants, insects — that form broad, visually coherent superordinate groups. The 1000-way-advantaged concepts tend to be man-made objects with fine within-category visual distinctions (kitchen items, grooming tools, containers) where fine-grained classification pressure forces the network to learn subtle feature differences.

---

## Figure S15. Top principal component pole images for each PCA source model

![Figure S15](figures/supplementary/supp_s15_pc_poles.png)

**Figure S15. The principal components used for coarse-graining capture interpretable visual and semantic axes.** For each of two PCA source models — AlexNet (left) and CLIP ViT-L/14 (right) — we show the 5 most-activating and 5 least-activating ImageNet images along PC1 through PC6, with variance explained (%) for each component. These PCs define the binary splits used to generate coarse labels: PC1 produces the 2-class partition, PCs 1–2 produce the 4-class partition, and so on. For AlexNet, PC1 separates broadly gray/man-made objects from colorful natural scenes. For CLIP, the axes are more semantically structured: PC1 separates animals/nature from artifacts/text. Despite these differences, models trained on labels from both sources achieve comparable brain alignment, suggesting that the *granularity* of supervision matters more than the specific axes along which categories are defined.

---

## Figure S16. Levels evaluation: hierarchical similarity benchmark

![Figure S16](figures/supplementary/supp_s16_levels.png)

**Figure S16. Coarse models improve on the Levels hierarchical similarity benchmark, particularly for between-class triplets.** Results on the Levels dataset (Muttenthaler et al., 2025), which evaluates model representations against human similarity judgments structured at multiple taxonomic levels. Three metrics are shown (rows): odd-one-out accuracy, uncertainty alignment, and triplet RSA. Three triplet types are shown (columns): within-class, class-boundary, and between-class. Coarse models show clear improvements on between-class and class-boundary triplets, where broad categorical structure determines the correct response. For within-class triplets, performance converges toward the 1000-way baseline at higher granularity levels (32–64 classes). This pattern is consistent with the THINGS results (Figure 4): coarse training enhances the between-category representational structure that dominates human similarity judgments at the superordinate level.

---

## Figure S17. Seed variability across benchmarks

![Figure S17](figures/supplementary/supp_s17_seed_variability.png)

**Figure S17. Alignment scores are highly stable across random training seeds.** Individual seed scores (colored markers: circle = seed 1, square = seed 2, triangle = seed 3) with 95% bootstrap CIs (error bars) for the 1000-class baseline and three CLIP coarse-grained models (8, 16, 32 classes) across three high-level alignment benchmarks: macaque IT electrophysiology **(A)**, human ventral visual stream fMRI **(B)**, and THINGS behavioral similarity **(C)**. Black horizontal lines indicate the cross-seed mean. Across all conditions and benchmarks, seed-to-seed variability is substantially smaller than within-seed bootstrap uncertainty, confirming that training stochasticity is not a dominant source of variance in the alignment results. The largest seed spread occurs for TVSD IT **(A)**, which is expected given the limited test data (2 monkeys, ~100 stimuli). For THINGS **(C)**, both the scores and CIs are tightly clustered, reflecting the large evaluation set (~1,480 concepts). These results complement Figure S9 by isolating the effect of random initialization from inter-subject variability.

---

*All supplementary figure scripts are available in `manuscript/figures/supplementary/` and can be regenerated from the project root using the commands listed in the supplementary README.*
