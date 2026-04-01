# Carving nature at its joints: a coarse feedback signal for learning human-aligned visual representations

---

## Abstract

A long-standing goal across neuroscience, cognitive science, and AI is to build artificial neural networks that process information like the brain. A prevailing assumption has been that achieving this requires increasingly rich training signals — driving the field from supervised classification over 1,000 categories toward self-supervised and single-image objectives that capture ever more fine-grained structure. Here, we start by exploring the opposite direction: how coarse can the training signal be while still producing representations that match how humans perceive and organize the world? We developed a data-driven coarse-graining method that progressively partitions images into broad categories (2, 4, 8, 16, …) without manual annotations. We trained hundreds of neural networks on ImageNet with these coarse supervisory signals. Strikingly, models trained with just a handful of broad categories achieved the highest alignment with human perceptual judgments of any neural network tested — outperforming both fine-grained supervised models and large-scale pretrained systems across every architecture we evaluated. These coarsely trained models also develop strong alignment with neural responses in both macaque and human visual cortex, using orders of magnitude fewer categories than standard benchmarks. The advantage grows stronger in low-data regimes: coarse feedback yields high alignment even with limited training samples. This work shows coarse feedback is a surprisingly powerful learning signal for building human-aligned artificial neural networks.

---

**Status:** All updated figures from Fig. 1 to Fig. 6 are complete. Supplementary figures are maintained separately in `manuscript/figures/supplementary/`.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Paper Narrative

### Introduction

DNNs trained on fine-grained classification (e.g., 1,000-category ImageNet) yield representations surprisingly well-aligned with primate visual cortex. This catalyzed the neuroAI field toward increasingly rich training signals — from hundreds of supervised categories to self-supervised and single-image objectives. The prevailing assumption: richness in the learning signal drives brain-like representations.

Yet this sits in tension with biological development — children initially carve the visual world into broad, coarse categories, only gradually acquiring finer distinctions. **Central question:** how coarse can the supervisory signal be while still producing representations that align with human perception and neural processing?

We explore the opposite direction from the field's trajectory. We developed a data-driven coarse-graining method (recursive PCA median splits → 2, 4, 8, 16, … classes) and trained hundreds of networks on ImageNet with these coarse signals — holding dataset, architecture, and all hyperparameters constant. This isolates the effect of supervisory signal granularity, evaluated against human behavioral judgments, macaque single-neuron recordings, and human fMRI.

### Section: Coarse-graining the input space (Fig. 1)

Neural representations in visual cortex are organized along a few dominant axes of variation (animacy, real-world size, etc.). We reasoned that principal axes of a learned visual representation could serve as a natural basis for partitioning images at multiple scales. Extract latent representations from a pretrained model (CLIP), perform PCA. Median split along PC1 → 2 classes (recovers animate–inanimate without manual labeling). Split each group along PC2 → 4 classes. Continue recursively → binary tree of 8, 16, 32, … classes. Not dependent on choice of pretrained model — qualitatively identical splits from AlexNet, ViT, DINOv3 (Supplementary Fig. X).

### Section: Coarse training produces fundamentally different representations (Fig. 2)

A network receiving a 4-way signal learns a sharply categorical representational space — four classes pulled cleanly apart. Critically, these are *not* a smoothed or low-dimensional version of what a 1,000-way classifier learns. They reflect a qualitatively different geometry that cannot be recovered by projecting the fine-grained model's representations into fewer dimensions. The coarseness of the learning signal fundamentally alters the structure of the learned representation.

### Section: Measuring biological alignment (methods)

RSA provides a common currency across systems. For N stimuli, compute pairwise distances → N × N representational similarity matrix. This depends only on relational structure, making it comparable across modalities (network activations, neural recordings, behavioral judgments). Alignment = correlation between RSMs from model and biological system.

### Section: A coarse feedback signal is sufficient for high neural alignment (Fig. 3)

Based on prevailing assumptions, reducing categories by orders of magnitude should substantially degrade neural alignment. We evaluated against: (1) NSD — whole-brain fMRI from 8 human subjects viewing thousands of natural images; (2) TVSD — single-neuron spiking responses from macaque IT to object images.

Across both datasets and both scales of measurement, coarsely trained networks achieved alignment comparable to, and in several cases exceeding, fine-grained supervised models. However, not any coarse partitioning suffices — pixel-based categories (luminance, contrast) show no meaningful neural alignment. What matters is that categories reflect the natural structure of visual experience.

### Section: Coarse feedback produces the most behaviorally aligned model (Fig. 4)

THINGS dataset: 4.7M odd-one-out triplet judgments from 12K+ participants → 66-dimensional behavioral embedding capturing human object perception. Networks trained on coarse signals did not merely approach fine-grained models — they *substantially exceeded* them. Benchmarked against pretrained models spanning diverse architectures and training paradigms; coarsely trained networks achieved the highest behavioral alignment of any model tested.

The fine-grained training signal appears to work *against* the broad categorical distinctions that dominate human perception.

### Section: Coarse feedback achieves high alignment even with limited data (Fig. 4D)

Models trained with coarse feedback on ~1% of ImageNet achieved higher behavioral alignment than a 1,000-class model trained on the full 1.2M images. Coarse supervision produces more human-aligned representations with dramatically less data — the relevant structure in visual experience may be learnable from sparse input when the objective is matched to the scale of perceptual organization.

### Section: Coarse training improves alignment across all semantic categories (Fig. 5)

Per-concept decomposition: for each of ~1,850 THINGS concepts, compute per-concept RSA contribution. Across *every* semantic category, the coarse model achieves higher alignment than the fine-grained model. The advantage is not confined to broad, easily separable domains — it extends to categories with fine-grained internal structure. The coarse model's RDM more faithfully mirrors the block-diagonal organization of human perception.

### Section: Coarse training generalizes across modern architectures (Fig. 6)

Trained ResNet-50, ConvNeXt, and ViT from scratch across the full coarse-to-fine spectrum. The pattern holds: coarse feedback produces representations with high alignment to human perceptual judgments, matching or exceeding 1,000-class counterparts. The advantage is not an artifact of limited model capacity — it emerges consistently regardless of architectural family, depth, or inductive bias.

### Discussion

Key claim: a coarse learning signal that partitions the input space into broad categories is *sufficient* to give rise to rich, human-aligned internal representations. These models develop representational structure that captures fine-grained perceptual distinctions despite never being asked to make them. This reframes what makes a good training objective for models of biological vision — less about how much information the signal carries, more about whether it carves the input space along perceptually relevant boundaries.

The coarse-graining procedure (recursive PCA median splits) is effective but not uniquely so — large-scale pretrained networks are not necessary, and median split is one of many possible criteria. Characterizing what properties a coarse signal must have to be effective is a natural next step.

The framework is in principle agnostic to input modality. Whether analogous coarse signals improve alignment in other sensory domains or language remains to be tested. Core finding: the computational principles underlying human-aligned visual representations may be far simpler than assumed.

---

## Figure-by-Figure Plan

The six main figures follow a progression: **schematic → representation analysis → neural data → behavioral data → per-concept analysis → architecture generalization**.

1. **Figure 1** — Schematic overview of the method and experimental pipeline.
2. **Figure 2** — Categorical nature of representations: PCA-based coarse-graining procedure (shared-PCA scatter) + learned representations are qualitatively different (model activation PCs).
3. **Figure 3** — Neural alignment across species: macaque electrophysiology (TVSD) and human fMRI (NSD), coarseness curves (raw Spearman rho). Per-layer profiles in supplementary.
4. **Figure 4** — Behavioral alignment (THINGS): coarseness, model comparison with pretrained baselines, data efficiency, and PC scatter panels. Per-concept analysis in Figure 5.
5. **Figure 5** — Per-concept alignment: category-sorted RDMs showing *why* coarse models win, per-concept scatter, and advantage histogram.
6. **Figure 6** — Architecture generalization: coarseness finding holds across ResNet-50, ConvNeXt, and ViT-B/16.

---

## Figure 1: Schematic + Label Space

**Directory:** `manuscript/figures/fig1/`

**Narrative role:** Schematic overview of the method and experimental pipeline. Panel 1a shows the label space PCA scatter illustrating the coarse-graining procedure.

### Panel 1a -- Label space PCA scatter (figure1a)

Single row, 4 columns: 1000-way (wider) | thin divider | 2-way | 4-way. Three PC1 vs PC2 scatter plots of ImageNet images (1 per class, 1000 points), projected onto the same CLIP PCA axes. All three share identical coordinates -- only the coloring changes:

```
+------------------+---+---------------+---------------+
| 1000-way colored |div| 2-way colored | 4-way colored |
| (1.15× width)    | | | (1× width)    | (1× width)    |
+------------------+---+---------------+---------------+
```

- **1000-way:** Each ImageNet class gets a unique color (tab20 cycling with jitter) -> continuous gradient, no structure.
- **2-way:** Median split on PC1 -> 2 colors (teal #1b9e77 / orange #d95f02).
- **4-way:** Median splits on PC1 then PC2 -> 4 colors (dark green #2d6a4f, light green #74c69d, gold #e8963e, red #d64045).

Six representative ImageNet images shown as 75×75 thumbnail insets with colored borders and connector arrows. Position repulsion algorithm avoids overlap.

- *Key message:* The coarse-graining procedure is simple and principled -- median splits along PCA axes of a pretrained feature space. Same images, same coordinates, different label assignments.
- *Script:* `manuscript/figures/fig1/plot_label_space.py` (with `--recompute` to regenerate PCA data)
- *Data:* `manuscript/figures/fig2/pc_scatter_1per_class.npz` (shared cache)
- *Output:* `figure1a.png` + `figure1a.svg` (14.8" × 4.8", 300 DPI)

---

## Figure 2: Categorical Nature of Representations

**Directory:** `manuscript/figures/fig2/`

**Narrative role:** Visual demonstration that coarse-trained representations are qualitatively different. The *learned* representations reorganize geometry in fundamentally different ways depending on label granularity -- different geometry, not just reduced geometry.

### Layout -- 1 row × 2 columns (image mosaic)

```
+--------------------------------------+--------------------------------------+
| a: CNN trained on 1,000 classes      | b: CNN trained on 4 coarse classes   |
| (FC1 PCs, image mosaic, 4-way col.) | (FC1 PCs, image mosaic, 4-way col.) |
+--------------------------------------+--------------------------------------+
```

Two side-by-side PC1 vs PC2 **image mosaics** of ImageNet activations (FC1 layer, L2-normalized). 1,000 randomly sampled images (seed=42) placed as 96×96 thumbnails at their actual PCA coordinates. Points colored by 4-way coarse labels (dark green #1B7A4F, bright teal #50C888, orange #E88A2A, red #D63540). Each panel has a white-background inset scatter (top-left, 24% × 24%) showing the full 50,000-point distribution.

- **Panel a (1000-way model):** Smooth gradient -- 4-way coarse labels are intermixed with no clear boundaries. The model distributes representations to separate 1000 classes, destroying the broad categorical structure.
- **Panel b (4-way model):** Four well-separated clusters, each internally variable but categorically distinct.

- *Key visual:* Direct comparison of how training objective reshapes learned geometry. The 1000-way model shows a continuous gradient; the 4-way model imposes clear categorical boundaries.
- *Key message:* Coarse training fundamentally changes how the network organizes visual information -- different geometry, not just reduced geometry. These representations cannot be recovered by dimensionality reduction of a fine-grained model.
- *Script:* `manuscript/figures/fig2/figure2.py` → calls `plot_representations.py`
- *Data source:* `experiments/representation_analysis/2pcs_compare/data_4way_alexnet.npz` (AlexNet-PCA labels, imagenet-mini-50, L2-normed FC1).
- *Output:* `figure2.png` (14" × 6", 300 DPI)

### Observed results

**(a) 1000-way model learned representations.** Smooth gradient in PC space -- coarse labels are intermixed with no clear boundaries. The model distributes representations to separate 1000 classes, destroying the broad categorical structure.

**(b) 4-way model learned representations.** Four well-separated clusters, each internally variable but categorically distinct. Image thumbnails reveal the semantic content of each cluster.

**Cross-model RSA (described in text, supplementary).** Cross-model RSA (1000-way vs. coarse) increases from rho ~ 0.22 (2-way) to rho ~ 0.52 (64-way), but never approaches the inter-seed baseline (rho ~ 0.76). Projection control: projected-1K vs. coarse RSA is extremely low (rho ~ 0.03 at 2-class, rho ~ 0.35 at 64-class), confirming coarse features cannot be recovered by PCA of the fine-grained model.

---

## Figure 3: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present neural alignment results from both macaque electrophysiology (TVSD) and human fMRI (NSD) in a single unified figure. This cross-species layout immediately demonstrates that the coarseness finding is robust -- it holds in spiking data and BOLD fMRI, across early and late visual regions, and across two completely different species.

### Layout -- 2 rows x 3 columns (14" × 8.5")

Each dataset occupies **one row**. Column 0 has dataset schematics; columns 1-2 have coarseness data panels. Column headers: "Early Visual Cortex" and "Higher Visual Cortex". Per-layer profiles moved to supplementary.

```
+------------------+-------------------------+-------------------------+
| A: TVSD          | B: TVSD V1 (Early)      | C: TVSD IT (Higher)     |
|   schematic      |   [lollipop strip 14%]  |   [lollipop strip 14%]  |
|   (objects →     |   [scatter plot   86%]  |   [scatter plot   86%]  |
|    monkey)       |                         |                         |
+------------------+-------------------------+-------------------------+
| D: NSD           | E: NSD Early Visual     | F: NSD Ventral Visual   |
|   schematic      |   [lollipop strip 14%]  |   [lollipop strip 14%]  |
|   (scenes →      |   [scatter plot   86%]  |   [scatter plot   86%]  |
|    human/fMRI)   |                         |                         |
+------------------+-------------------------+-------------------------+
```

**2 rows x 3 columns = 6 panels (A-F).** Each data cell (B, C, E, F) contains an inner 2-panel grid: a thin lollipop strip on top + scatter plot below.

### Lollipop strips (top of each data cell)

Horizontal lollipop chart showing the **minimum number of training classes (k*) needed to match the 1000-way baseline CI**, for AlexNet and CLIP only (no Pixels). Each architecture gets one row with a colored stem + marker at k*. Vertical dashed orange line at the 1000-way position. Gray background with interlocking x-axis break matching the scatter below.

- *Y-axis:* Two rows (AlexNet bottom, CLIP top), labeled with architecture names.
- *X-axis:* Log2 scale, shared with scatter below (no separate labels).

### Coarseness scatter plots (bottom of each data cell -- raw Spearman rho)

Y-axis shows raw Spearman rho. Log2 x-axis (2 -> 1000) with axis break before the 1000-way bar. Three PCA source models (AlexNet, CLIP, Pixels) as separate markers/colors, plus 1000-way baseline (orange dashed line + diamond) and untrained baseline (gray dashed line, labeled on bottom row only).

**Color scheme:** AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (muted tan `#c0a898`, triangle-down), 1K (warm amber `#e8963e`, diamond). Untrained (gray `#999999`).

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
