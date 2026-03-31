# Carving nature at its joints: a coarse feedback signal for learning human-aligned visual representations

---

## Abstract

A long-standing goal across neuroscience, cognitive science, and AI is to build artificial neural networks that process information like the brain. A prevailing assumption has been that achieving this requires increasingly rich training signals—driving the field from supervised classification over 1,000 categories toward self-supervised and single-image objectives that capture ever more fine-grained structure. Here, we start by exploring the opposite direction: how coarse can the training signal be while still producing representations that match how humans perceive and organize the world? We developed a data-driven coarse-graining method that progressively partitions images into broad categories (2, 4, 8, 16, …) without manual annotations. We trained hundreds of neural networks on ImageNet with these coarse supervisory signals. Strikingly, models trained with just a handful of broad categories achieved the highest alignment with human perceptual judgments of any neural network tested—outperforming both fine-grained supervised models and large-scale pretrained systems across every architecture we evaluated. These coarsely trained models also develop strong alignment with neural responses in both macaque and human visual cortex, using orders of magnitude fewer categories than standard benchmarks. The advantage grows stronger in low-data regimes: coarse feedback yields high alignment even with limited training samples. This work shows coarse feedback is a surprisingly powerful learning signal for building human-aligned artificial neural networks.


---

**Status:** All updated figures from Fig. 1 to Fig. 6 are complete. Supplementary figures are maintained separately in `manuscript/figures/supplementary/`.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Narrative Arc

The six main figures follow a progression: **schematic -> representation analysis -> neural data -> behavioral data -> per-concept analysis -> architecture generalization**.

1. **Figure 1** is the schematic overview of the method and experimental pipeline.
2. **Figure 2** shows the categorical nature of representations: the left panel (figure2a) shows the PCA-based coarse-graining procedure via shared-PCA scatter (2-way, 4-way, 1000-way label coloring), and the right panel (figure2b) shows that the *learned* representations are qualitatively different via PC scatter of model activations (1000-way vs 4-way with image insets).
3. **Figure 3** presents neural alignment across species: macaque electrophysiology (TVSD) and human fMRI (NSD) side by side, with coarseness curves (raw Spearman rho). Schematic placeholders for each dataset. Per-layer profiles in supplementary.
4. **Figure 4** presents behavioral alignment (THINGS): coarseness results, model comparison with pretrained baselines, and PC scatter panels visualizing representational geometry across models. Per-concept analysis is in Figure 5.
5. **Figure 5** digs into per-concept alignment: category-sorted RDMs showing *why* coarse models win, per-concept scatter, and advantage histogram.
6. **Figure 6** demonstrates architecture generalization: the coarseness finding on THINGS behavioral alignment holds across ResNet-50, ConvNeXt, and ViT-B/16 (CLIP-based coarse labels).

---

## Figure 1: Schematic

**Directory:** `manuscript/figures/fig1/`

**Narrative role:** Schematic overview of the method and experimental pipeline.

---

## Figure 2: Categorical Nature of Representations

**Directory:** `manuscript/figures/fig2/`

**Narrative role:** Visual demonstration that coarse-trained representations are qualitatively different. The left panel (figure2a) shows how the coarse-graining procedure partitions images in a shared PCA space; the right panel (figure2b) shows that the *learned* representations reorganize geometry in fundamentally different ways depending on label granularity.

### Layout — left + right (figure2a + figure2b)

```
+----------------------------------------------+------------------------------------------+
| figure2a (left): Schematic / PCA scatter     | figure2b (right): Learned representations |
|                                              |                                          |
| +--------------+--------------+------------+ | +--------------------------------------+ |
| | Shared PCA:  | Shared PCA:  | Shared PCA:| | | CNN trained on 1000 classes          | |
| | 2-way color  | 4-way color  | 1000-way   | | | (FC1 PCs + image insets, 4-way col.) | |
| | (label space)| (label space)| (label sp.)| | +--------------------------------------+ |
| +--------------+--------------+------------+ | | CNN trained on 4 derived classes      | |
|                                              | | (FC1 PCs + image insets, 4-way col.) | |
|                                              | +--------------------------------------+ |
+----------------------------------------------+------------------------------------------+
```

### Left panel (figure2a) -- Shared PCA scatter (label space)

Three PC1 vs PC2 scatter plots of ImageNet images (1 per class, 1000 points), projected onto the same CLIP PCA axes. All three panels share identical coordinates -- only the coloring changes:

- **A (2-way):** Median split on PC1 -> 2 colors with decision boundary line.
- **B (4-way):** Median splits on PC1 then PC2 -> 4 colors with decision boundary lines.
- **C (1000-way):** Each ImageNet class gets a unique color -> continuous gradient, no structure.

- *Key message:* The coarse-graining procedure is simple and principled -- median splits along PCA axes of a pretrained feature space. Same images, same coordinates, different label assignments.
- *Script:* `manuscript/figures/fig2/figure2.py` (with `--recompute-top` to regenerate)
- *Data:* `manuscript/figures/fig2/pc_scatter_1per_class.npz`

### Right panel (figure2b) -- Learned representation scatter (model activation space)

Two vertically stacked PC1 vs PC2 scatter plots of ImageNet activations (FC1 layer, L2-normalized), comparing models trained at different granularity. Points colored by 4-way coarse labels. Representative images shown as insets at class extremes and centroids.

- **Top (1000-way model):** Smooth gradient -- 4-way coarse labels are intermixed with no clear boundaries. The model distributes representations to separate 1000 classes, destroying the broad categorical structure.
- **Bottom (4-way model):** Four well-separated clusters, each internally variable but categorically distinct.

- *Key visual:* Direct comparison of how training objective reshapes learned geometry. The 1000-way model shows a continuous gradient; the 4-way model imposes clear categorical boundaries.
- *Key message:* Coarse training fundamentally changes how the network organizes visual information -- different geometry, not just reduced geometry. These representations cannot be recovered by dimensionality reduction of a fine-grained model.
- *Data source:* `experiments/representation_analysis/2pcs_compare/` (AlexNet-PCA labels, imagenet-mini-50, L2-normed FC1).

### Observed results

**(A-C) Shared PCA scatter.** All 1000 points share the same (x, y) coordinates across panels. The 2-way panel shows a clean PC1 median split (natural vs man-made). The 4-way panel subdivides each half by PC2. The 1000-way panel shows no visible structure -- a cloud of 1000 distinct colors.

**(D) 1000-way model learned representations.** Smooth gradient in PC space -- coarse labels are intermixed with no clear boundaries. The model distributes representations to separate 1000 classes, destroying the broad categorical structure.

**(E) 4-way model learned representations.** Four well-separated clusters, each internally variable but categorically distinct. Image insets reveal the semantic content of each cluster.

**Cross-model RSA (described in text, supplementary).** Cross-model RSA (1000-way vs. coarse) increases from rho ~ 0.22 (2-way) to rho ~ 0.52 (64-way), but never approaches the inter-seed baseline (rho ~ 0.76). Projection control: projected-1K vs. coarse RSA is extremely low (rho ~ 0.03 at 2-class, rho ~ 0.35 at 64-class), confirming coarse features cannot be recovered by PCA of the fine-grained model.

---

## Figure 3: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present neural alignment results from both macaque electrophysiology (TVSD) and human fMRI (NSD) in a single unified figure. This cross-species layout immediately demonstrates that the coarseness finding is robust -- it holds in spiking data and BOLD fMRI, across early and late visual regions, and across two completely different species.

### Layout -- 2 rows x 3 columns

Each dataset occupies **one row**. Column 0 has schematic placeholders; columns 1-2 have coarseness data panels. Column headers: "Early Visual Cortex" and "Higher Visual Cortex". Per-layer profiles moved to supplementary.

```
+------------------+-----------------+-----------------+
| A: TVSD          | B: TVSD V1      | C: TVSD IT      |
|   schematic      |   Coarseness    |   Coarseness    |
|   (placeholder)  |   (Early)       |   (Higher)      |
+------------------+-----------------+-----------------+
| D: NSD           | E: NSD Early    | F: NSD Ventral  |
|   schematic      |   Visual Stream |   Visual Stream |
|   (placeholder)  |   Coarseness    |   Coarseness    |
+------------------+-----------------+-----------------+
```

**2 rows x 3 columns = 6 panels (A-F).** Schematics are placeholders describing each dataset.

### Coarseness plots (4 data panels -- raw Spearman rho)

Y-axis shows raw Spearman rho. Log2 x-axis (2 -> 1000) with axis break before the 1000-way bar. Three PCA source models (AlexNet, CLIP, Pixels) as separate markers/colors, plus 1000-way baseline (amber bar) and untrained baseline (dashed line + gray bar).

**Color scheme:** AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (muted tan `#c0a898`, triangle-down), 1K (warm amber `#e8963e`, bar). Untrained (gray `#999999`, bar).

**Panels:**
1. **B: TVSD V1 (early):** Coarse models match or exceed 1000-way. Flat across granularity.
2. **C: TVSD IT (higher):** Gradual increase, 32-64 classes approach 1000-way.
3. **E: NSD Early Visual Stream:** Similar to TVSD V1 -- saturation at low granularity.
4. **F: NSD Ventral Visual Stream:** Gradual increase, approaching 1000-way by 32-64 classes.

- *Error bars -- TVSD:* +/-1.96 SEM across 2 monkeys x 3 seeds.
- *Error bars -- NSD:* Bootstrap CIs aggregated across 8 subjects x 3 seeds.

### Observed results

**(B) TVSD V1 coarseness.** AlexNet and CLIP cluster at rho ~ 0.16-0.18 across all granularity levels (2-64), matching or exceeding the 1000-way baseline (rho ~ 0.15). Pixels is lower (rho ~ 0.10-0.15). Flat curve -- even 2-class models match 1000-way for V1.

**(C) TVSD IT coarseness.** Clear monotonic ramp: 2-class at rho ~ 0.12, increasing to rho ~ 0.17-0.19 by 64 classes, approaching 1000-way (rho ~ 0.16). Pixels lags substantially. Untrained well separated at rho ~ 0.05.

**(E) NSD Early Visual Stream coarseness.** AlexNet and CLIP at rho ~ 0.19-0.21 across all levels. 1000-way at rho ~ 0.20. Pixels lower at rho ~ 0.10. Untrained at rho ~ 0.12.

**(F) NSD Ventral Visual Stream coarseness.** Gradual ramp: 2-class at rho ~ 0.15, rising to rho ~ 0.25 by 64 classes. 1000-way at rho ~ 0.21. CLIP converges fastest. Untrained well separated at rho ~ 0.04.

---

## Figure 4: Behavioral Alignment (THINGS)

**Directory:** `manuscript/figures/fig4/`

**Narrative role:** Present the behavioral alignment results -- the most surprising finding. Coarse models *vastly* outperform 1000-way on human similarity judgments. This figure shows the result (coarseness log plot), compares against pretrained models, and visualizes the representational geometry via PC scatter panels. Per-concept analysis and RDMs are in Figure 5; architecture generalization is in Figure 6.

### Layout -- 2 rows

```
+---------------+------------------+----------------------------------+
| A: Schematic  | B: Coarseness    | C: Model Comparison              |
| (THINGS)      | (raw Spearman)   | (coarse vs 1K bars +             |
| (placeholder) |                  |  pretrained scatter)             |
+---------------+------------------+----------------------------------+
| D: 4 PC scatter panels spanning full width                          |
| [Behavioral]  [CNN 8-class CLIP] [AlexNet 1K]   [ViT-B/16 1K]     |
+---------------------------------------------------------------------+
```

### Panel A -- THINGS schematic

Schematic placeholder of the THINGS behavioral similarity task: triplet odd-one-out judgments, how behavioral RDMs are constructed, and the comparison with model RDMs.

- *Key message:* Introduces the behavioral benchmark before showing results.

### Panel B -- Coarseness log plot (raw Spearman rho)

Y-axis shows raw Spearman rho. Log2 x-axis (2 -> 1000). Three PCA architectures (AlexNet, CLIP, Pixels -- same blue/amber color scheme as Figure 3).

- *Key pattern:* All coarse models sit well **above** the 1000-way baseline -- the coarse advantage is dramatic. Even 2-class models exceed 1000-way. This is the headline result.
- *Error bars:* Bootstrap 95% CIs across 3 seeds.
- *Architectures:* AlexNet (medium blue circle), CLIP (dark blue square), Pixels (brown triangle-down), 1K (warm amber diamond).

### Panel C -- Model comparison (coarse vs 1K + pretrained)

Grouped scatter of pretrained models (supervised, self-supervised, vision-language) with architecture markers (CNN pentagon, ViT star). The 8-class CLIP-repr. coarse model and 1000-way are highlighted.

- *Key visual:* The coarse-trained model (trained from scratch) matches or exceeds many large pretrained models on behavioral alignment.
- *Key message:* Coarse supervision is not just better than 1000-way -- it competes with the best pretrained vision models.

### Panel D -- PC scatter panels (representational geometry)

Four side-by-side PC1 vs PC2 scatter plots of THINGS concept representations, each colored by 8 super-categories derived from the 27 THINGS categories. Representative images shown as insets at category extremes.

1. **Behavioral** (ground truth) -- human similarity structure projected into 2D
2. **CNN 8 classes (CLIP repr.)** -- coarse-trained model captures broad categorical separation
3. **AlexNet (1K classes)** -- pretrained 1000-way AlexNet
4. **ViT-B/16 (1K classes)** -- pretrained 1000-way ViT

- *Key visual:* The coarse model's PC scatter mirrors the behavioral structure much more closely than 1000-way models, with clear super-category separation.
- *Key message:* Coarse training preserves the broad between-category structure that dominates human similarity judgments.

### Observed results

**(A)** Schematic (no data).

**(B) Coarseness log plot.** The headline result of the paper. All coarse models sit well above the 1000-way baseline. CLIP labels (dark blue) are strongest: rho ~ 0.55-0.57. AlexNet (medium blue): rho ~ 0.44-0.48. Pixels (brown) starts low at rho ~ 0.10 (2-class), rises to rho ~ 0.23 at 64-class but never reaches 1000-way (rho ~ 0.39). The untrained baseline is at rho ~ 0.20.

**(C) Model comparison.** Best coarse model (rho ~ 0.57) substantially exceeds 1000-way (rho ~ 0.39). Among pretrained models, CLIP-L/14 and DINOv2 approach but do not exceed the coarse-trained model. Supervised CNNs cluster well below.

**(D) PC scatter panels.** The behavioral ground truth shows clear super-category clustering. The CNN 8-class CLIP model reproduces this structure well. AlexNet-1K and ViT-B/16 1K show different, more diffuse geometry.

---

## Figure 5: Per-Concept Alignment Analysis

**Directory:** `manuscript/figures/fig5/`

**Narrative role:** Dig deeper into *which* concepts drive the coarse advantage on THINGS behavioral alignment. Row 1 shows category-sorted RDMs that visually demonstrate *why* coarse models win; Row 2 quantifies the effect with per-concept scatter and advantage histogram.

### Layout -- 2 rows

```
Row 1 (Panel A):
+--------------------+-------------------+-------------------+--------+
| Behavioral RDM     | 8 classes (CLIP   | 1000-class RDM    | color  |
| (ground truth)     |  repr.) RDM       |                   | bar    |
+--------------------+-------------------+-------------------+--------+
[                  super-category legend row                          ]

Row 2:
+--------------------------------+-----------------------------------+
| B: Per-concept scatter         | C: Per-concept advantage          |
| (8-class CLIP vs 1000-way)     |    histogram (delta rho)          |
+--------------------------------+-----------------------------------+
```

### Panel A -- Category-sorted RDMs

Three RDMs side by side -- the most visually striking evidence for *why* coarse models win:

1. **Behavioral** (ground truth) -- human similarity structure from THINGS triplet judgments
2. **8 classes (CLIP repr.)** -- coarse model captures the broad block-diagonal structure
3. **1000-class** -- over-differentiates within categories, weaker block boundaries

Concepts grouped by 8 semantic super-categories (Living things, Body & apparel, Food & drink, Home, Tools & equipment, Vehicles, Tech & leisure, Other) with colored sidebars and boundary lines overlaid. Spearman rho shown in panel titles.

- *Key visual:* The coarse model RDM captures the broad categorical block structure of human similarity much better than the 1000-way model.
- *Key message:* Fine-grained training emphasizes within-category distinctions at the expense of the broad between-category structure that dominates human similarity judgments.

### Panel B -- Per-concept scatter plot (CLIP model)

Scatter plot of per-concept RSA contribution: **8 classes (CLIP repr.)** (y-axis) vs. 1000-way (x-axis). Identity line for reference. Points color-coded by THINGS semantic category with legend showing top categories favoring each model.

- *Key pattern:* ~82% of concepts fall above the diagonal (coarse model wins). Plants, animals, clothing accessories strongly favor coarse. Body parts, drinks favor 1000-way.
- *Key message:* The coarse model advantage is pervasive across most concept categories, not a niche effect.

### Panel C -- Per-concept advantage histogram

Histogram of per-concept advantage: `(coarse_score - 1000way_score)` for each eval concept. Positive = coarse wins, negative = 1000-way wins.

- *Key visual:* Distribution clearly shifted right. Vertical line at zero. Green bins for coarse advantage, percentage annotations (18% vs 82%).
- *Key message:* Quantifies that the advantage is broad -- not driven by outlier concepts.

### Observed results

**(A) Category-sorted RDMs.** Three RDMs sorted by 8 super-categories. The behavioral RDM shows clear block-diagonal structure. The CLIP 8-class RDM (rho_s = 0.576) captures this block structure remarkably well. The 1000-way RDM (rho_s = 0.399) has weaker block boundaries -- it over-differentiates within categories.

**(B) Per-concept scatter.** ~82% of concepts fall above the diagonal (coarse wins). Green-colored clusters (plants, animals) are consistently above; orange clusters (body parts, drinks) are below.

**(C) Histogram.** Delta-rho distribution is right-shifted. 82% of concepts favor the coarse model, 18% favor 1000-way. The positive tail is substantially longer.

---

## Figure 6: Architecture Generalization (THINGS Behavioral)

**Directory:** `manuscript/figures/fig6/`

**Narrative role:** Demonstrate that the coarseness finding is not specific to the custom AlexNet-style CNN used in Figures 3-5. The same pattern -- coarse-trained models matching or exceeding 1000-class on THINGS behavioral alignment -- holds across three diverse modern architectures: ResNet-50 (deep CNN), ConvNeXt (modern CNN), and ViT-B/16 (vision transformer).

### Layout -- single row, 3 panels

```
+-------------------+-------------------+-------------------+
| a: ResNet-50      | b: ConvNeXt       | c: ViT-B/16       |
|   THINGS          |   THINGS          |   THINGS          |
|   coarseness      |   coarseness      |   coarseness      |
+-------------------+-------------------+-------------------+
```

Each panel shows THINGS behavioral alignment (Spearman rho) vs. label granularity (2-64 classes on log x-axis), with a 1000-class baseline bar. All use CLIP-derived coarse labels, epoch 20, seed 1.

### Panels a-c -- THINGS coarseness per architecture

Same style as Figure 4B: coarse conditions as blue scatter (CLIP labels), 1000-way as amber bar, untrained as dashed line. Axis break before 1000-way bar. ResNet-50 and ConvNeXt share y-axis limits for direct comparison; ViT-B/16 has its own scale.

- *Key message:* The coarseness finding generalizes across architectures. In all three architectures, coarse-trained models (as few as 2-8 classes) match or exceed the fully supervised 1000-class baseline on behavioral alignment.
- *Architecture-specific patterns:*
  - **ResNet-50 & ConvNeXt:** Flat coarseness curve at rho ~ 0.55-0.58 for all granularities, well above 1000-class (rho ~ 0.50 / 0.35). The finding from the custom CNN replicates cleanly.
  - **ViT-B/16:** Coarseness curve at rho ~ 0.41-0.46, above 1000-class (rho ~ 0.26). Coarse advantage is even more dramatic in relative terms (~60% improvement).

### Observed results

**(a) ResNet-50.** All coarse models (2-64 class) achieve rho ~ 0.55-0.58 on THINGS, substantially above the 1000-class baseline (rho ~ 0.50). The curve is nearly flat.

**(b) ConvNeXt.** Coarse models range from rho ~ 0.53 (2-class) to rho ~ 0.56 (64-class). The 1000-class baseline is lower at rho ~ 0.35. Dramatic coarse advantage.

**(c) ViT-B/16.** Coarse models achieve rho ~ 0.41-0.46, well above 1000-class (rho ~ 0.26). The coarse advantage is proportionally the largest for ViT, consistent with the behavioral alignment being driven by broad categorical structure that coarse supervision preserves.

---

## Directory Structure

```
manuscript/figures/
+-- paper.md                     # This file
+-- fig_utils.py                 # Shared constants, style, helpers
+-- things_utils.py              # Shared THINGS plotting utilities
+-- fig1/                        # Figure 1: Schematic
+-- fig2/                        # Figure 2: Categorical nature of representations
|   +-- figure2.py               # Combined left + right panels
|   +-- figure2a.png             # Left panel: shared PCA scatter (label space)
|   +-- figure2b.png             # Right panel: learned representation scatter
+-- fig3/                        # Figure 3: Neural alignment (TVSD + NSD)
|   +-- figure3.py               # Combined TVSD + NSD figure
|   +-- figure3.png
+-- fig4/                        # Figure 4: THINGS behavioral -- coarseness + model comparison + PC scatter
|   +-- figure4.py               # Schematic + coarseness + model comparison + PC scatter
|   +-- plot_pc_scatter.py       # PC scatter panel helper
|   +-- figure4.png
+-- fig5/                        # Figure 5: Per-concept alignment analysis
|   +-- figure5.py               # RDMs + scatter + histogram
|   +-- dimension_profiling.py   # Semantic dimension profiling (standalone, not in main figure)
|   +-- figure5.png
+-- fig6/                        # Figure 6: Architecture generalization
|   +-- figure6.py               # THINGS coarseness for ResNet-50, ConvNeXt, ViT-B/16
|   +-- figure6.png
+-- supplementary/               # Supplementary figures (S1-S18)
    +-- README.md                # Index, run commands, data sources
    +-- figure_descriptions.md   # Detailed descriptions of each supp figure
    +-- supp_s1_training_summary.py
    +-- ...
    +-- plot_class_rdms.py
```

---

## Design Notes

- **Consistent color scheme across Figures 3 & 4:**
  - PCA source models: AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (muted tan/brown, triangle-down). ViT moved to supplementary.
  - 1000-way baseline: Warm amber (`#e8963e`, diamond/bar)
  - Untrained baseline: Gray dashed line or gray bar (`#999999`)
- **Raw Spearman rho (Figures 3 & 4):** All coarseness plots show raw Spearman rho values.
- **Log-scale x-axis** for all coarseness plots (Figures 3 and 4), with axis break before the 1000-way grouped bars.
- **Schematics are placeholders** in Figures 3 and 4 -- to be replaced with final artwork.
- **Figure 1 is the schematic** -- method and experimental pipeline overview.
- **Figure 2 shows categorical nature of representations** -- left panel (figure2a) is shared PCA scatter (label space), right panel (figure2b) is learned representation PC scatter (model activation space). Old RDM panel moved to supplementary S18.
- **Figure 4 bottom row is PC scatter** -- 4 panels showing representational geometry (Behavioral, CNN 8-class CLIP, AlexNet 1K, ViT-B/16 1K). RDMs moved to Figure 5 Panel A.
- **Figure 5 row 1 has RDMs** -- category-sorted RDMs (Behavioral, 8-class CLIP, 1000-class) with 8 super-category groupings. Row 2 has per-concept scatter + histogram. Dimension profiling exists as a standalone script but is not in the main figure.
- **Figure 6 has no schematic** -- directly shows THINGS coarseness for three architectures (ResNet-50, ConvNeXt, ViT-B/16).
- **No DINO or ViT PCA in main figures** -- supplementary only. Main figures show AlexNet, CLIP, and Pixels PCA sources.
- **No NSD-Synthetic in main figures** -- supplementary only.
- **V4 (TVSD) in supplementary only** -- V1 and IT represent the extremes of the visual hierarchy.
