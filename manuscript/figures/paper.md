# Carving nature at its joints: a coarse feedback signal for learning human-aligned visual representations

---

## Abstract

Computational models of vision increasingly rely on detailed training objectives — from discriminating among thousands of object
  categories to self-supervised instance-level learning — under the assumption that finer-grained supervision yields more
  brain-like representations. Here we put this assumption to the test. We develop a data-driven method to continuously vary the
  granularity of a supervisory signal — from just 2 broad categories to 1,000 fine-grained classes — while holding architecture,
  data, and training procedure constant. We first show that coarse supervisory signals do not produce impoverished representations:
   they give rise to rich internal structure that is qualitatively distinct from fine-grained features and cannot be recovered by
  dimensionality reduction of a fully supervised model. We then measure alignment with biological vision across three measurement
  scales — macaque single-neuron recordings, human fMRI, and human behavioral similarity judgments — and find a striking pattern.
  In early visual cortex of both species, even the coarsest models match fine-grained alignment. Higher visual areas converge with
  only a few dozen categories. Most remarkably, on behavioral alignment, coarse-trained models dramatically exceed fine-grained
  training: a network distinguishing just 4 broad categories outperforms standard 1,000-way ImageNet classification by 40% and
  surpasses pretrained vision transformers and CLIP — because coarse supervision preserves the between-category structure that
  dominates how humans perceive similarity. These results reveal that learning to carve the world into broad categories gives rise
  to surprisingly rich representations that align with biological vision across scales, from single neurons to behavior. Beyond
  challenging the more-is-better intuition, these findings point to a general principle — that the categorical grain of supervision
   shapes representational geometry in ways that may inform the design of more biologically-aligned artificial neural networks.

---

**Status:** All updated figures from Fig. 1 to Fig. 5 are complete. Supplementary figures are maintained separately in `manuscript/figures/supplementary/`.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Narrative Arc

The five main figures follow a progression: **method overview + representation analysis -> neural data -> behavioral data -> per-concept analysis -> data efficiency**.

1. **Figure 1** combines the method overview with representation analysis: the left panel (figure1a) shows the PCA-based coarse-graining procedure via shared-PCA scatter (2-way, 4-way, 1000-way label coloring), and the right panel (figure1b) shows that the *learned* representations are qualitatively different via PC scatter of model activations (1000-way vs 4-way with image insets).
2. **Figure 2** presents neural alignment across species: macaque electrophysiology (TVSD) and human fMRI (NSD) side by side, with coarseness curves (raw Spearman rho). Schematic placeholders for each dataset. Per-layer profiles in supplementary.
3. **Figure 3** presents behavioral alignment (THINGS): coarseness results, model comparison with pretrained baselines, and PC scatter panels visualizing representational geometry across models. Per-concept analysis is in Figure 4.
4. **Figure 4** digs into per-concept alignment: category-sorted RDMs showing *why* coarse models win, per-concept scatter, and advantage histogram.
5. **Figure 5** introduces the data-efficiency paradigm: coarse vs fine-grained training at varying data scales across NSD (early + ventral) and THINGS.

---

## Figure 1: Method Overview + Representation Analysis

**Directory:** `manuscript/figures/fig1/`

**Narrative role:** Combines the method introduction with a visual demonstration that coarse-trained representations are qualitatively different. The left panel (figure1a) shows how the coarse-graining procedure partitions images in a shared PCA space; the right panel (figure1b) shows that the *learned* representations reorganize geometry in fundamentally different ways depending on label granularity.

### Layout — left + right (figure1a + figure1b)

```
+----------------------------------------------+------------------------------------------+
| figure1a (left): Schematic / PCA scatter     | figure1b (right): Learned representations |
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

### Left panel (figure1a) -- Shared PCA scatter (label space)

Three PC1 vs PC2 scatter plots of ImageNet images (1 per class, 1000 points), projected onto the same CLIP PCA axes. All three panels share identical coordinates -- only the coloring changes:

- **A (2-way):** Median split on PC1 -> 2 colors with decision boundary line.
- **B (4-way):** Median splits on PC1 then PC2 -> 4 colors with decision boundary lines.
- **C (1000-way):** Each ImageNet class gets a unique color -> continuous gradient, no structure.

- *Key message:* The coarse-graining procedure is simple and principled -- median splits along PCA axes of a pretrained feature space. Same images, same coordinates, different label assignments.
- *Script:* `manuscript/figures/fig1/figure1.py` (with `--recompute-top` to regenerate)
- *Data:* `manuscript/figures/fig1/pc_scatter_1per_class.npz`

### Right panel (figure1b) -- Learned representation scatter (model activation space)

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

## Figure 2: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig2/`

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

## Figure 3: Behavioral Alignment (THINGS)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present the behavioral alignment results -- the most surprising finding. Coarse models *vastly* outperform 1000-way on human similarity judgments. This figure shows the result (coarseness log plot), compares against pretrained models, and visualizes the representational geometry via PC scatter panels. Per-concept analysis and RDMs are in Figure 4; data efficiency is in Figure 5.

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

Y-axis shows raw Spearman rho. Log2 x-axis (2 -> 1000). Three PCA architectures (AlexNet, CLIP, Pixels -- same blue/amber color scheme as Figure 2).

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

## Figure 4: Per-Concept Alignment Analysis

**Directory:** `manuscript/figures/fig4/`

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

## Figure 5: Data Efficiency

**Directory:** `manuscript/figures/fig5/`

**Narrative role:** Introduce a new analysis paradigm -- varying the number of training images per class while holding granularity constant. Shows that coarse-trained models are more data-efficient than fine-grained models across both neural and behavioral alignment.

### Layout -- 2 panel groups (A + B)

```
+----------------------------------------------+-----------------------+
| A: Natural Scenes Dataset                    | B: THINGS Behavior    |
|  +-------------------+--------------------+  |                       |
|  | Early visual      | Ventral visual     |  | line plot             |
|  | stream            | stream             |  | (4 conditions x       |
|  | (line plot)       | (line plot)        |  |  4 data scales)       |
|  +-------------------+--------------------+  |                       |
+----------------------------------------------+-----------------------+
```

Panel A groups two NSD sub-panels under a shared "Natural Scenes Dataset" header with region subtitles. A vertical separator divides the NSD group from Panel B.

### Panel A -- NSD (Early Visual Stream + Ventral Visual Stream)

Two side-by-side line plots with 4 conditions (8, 16, 32, 1000-class) across 4 data scales (5K, 10K, 50K, 1.2M training images). Green shades for coarse models, orange for 1000-class.

- **Left sub-panel (Early visual stream):** Coarse and fine-grained models perform similarly, especially at larger data scales.
- **Right sub-panel (Ventral visual stream):** Coarse models consistently outperform 1000-class at all data scales, with the gap narrowing at full scale.
- *Key message:* Coarse training provides a better inductive bias when data is limited, even for neural alignment.

### Panel B -- THINGS Behavioral (line plot)

Same line-plot format as Panel A, showing behavioral alignment (Spearman rho) across data scales.

- *Key visual:* The coarse advantage is dramatic and persistent -- 1000-class never catches up, even at full ImageNet.
- *Key message:* Coarse training is not only better at full scale -- it is vastly more data-efficient for behavioral alignment.

### Observed results

**(A, left) NSD Early Visual Stream.** All conditions perform similarly at larger data scales. At 5K, coarse models (8, 16, 32-class) show a slight advantage. By 1.2M, all converge around rho ~ 0.19-0.21.

**(A, right) NSD Ventral Visual Stream.** All coarse models (8, 16, 32-class) outperform 1000-class across all data scales. At 5K, coarse models achieve rho ~ 0.10 while 1000-class is at rho ~ 0.05. All models converge toward rho ~ 0.25 at full scale (1.2M), but coarse models maintain a slight edge.

**(B) THINGS Behavioral.** The headline data-efficiency result. At 5K, 8-class achieves rho ~ 0.45 while 1000-class reaches only rho ~ 0.27. The gap persists at all scales. At full ImageNet (1.2M), coarse models reach rho ~ 0.57 vs 1000-class at rho ~ 0.40.

---

## Directory Structure

```
manuscript/figures/
+-- paper.md                     # This file
+-- fig_utils.py                 # Shared constants, style, helpers
+-- things_utils.py              # Shared THINGS plotting utilities
+-- fig1/                        # Figure 1: method overview + representation analysis
|   +-- figure1.py               # Combined left + right panels
|   +-- figure1a.png             # Left panel: shared PCA scatter (schematic)
|   +-- figure1b.png             # Right panel: learned representation scatter
+-- fig2/                        # Figure 2: Neural alignment (TVSD + NSD)
|   +-- figure2.py               # Combined TVSD + NSD figure
|   +-- figure2.png
+-- fig3/                        # Figure 3: THINGS behavioral -- coarseness + model comparison + PC scatter
|   +-- figure3.py               # Schematic + coarseness + model comparison + PC scatter
|   +-- plot_pc_scatter.py       # PC scatter panel helper
|   +-- figure3.png
+-- fig4/                        # Figure 4: Per-concept alignment analysis
|   +-- figure4.py               # RDMs + scatter + histogram
|   +-- dimension_profiling.py   # Semantic dimension profiling (standalone, not in main figure)
|   +-- figure4.png
+-- fig5/                        # Figure 5: Data efficiency
|   +-- figure5.py               # Data-efficiency line plots
|   +-- figure5.png
+-- supplementary/               # Supplementary figures (S1-S18)
    +-- README.md                # Index, run commands, data sources
    +-- figure_descriptions.md   # Detailed descriptions of each supp figure
    +-- supp_s1_training_summary.py
    +-- ...
    +-- plot_class_rdms.py
```

---

## Design Notes

- **Consistent color scheme across Figures 2 & 3:**
  - PCA source models: AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (muted tan/brown, triangle-down). ViT moved to supplementary.
  - 1000-way baseline: Warm amber (`#e8963e`, diamond/bar)
  - Untrained baseline: Gray dashed line or gray bar (`#999999`)
- **Raw Spearman rho (Figures 2 & 3):** All coarseness plots show raw Spearman rho values.
- **Log-scale x-axis** for all coarseness plots (Figures 2 and 3), with axis break before the 1000-way grouped bars.
- **Schematics are placeholders** in Figures 2 and 3 -- to be replaced with final artwork.
- **Figure 1 combines method + representation analysis** -- left panel (figure1a) is shared PCA scatter (label space), right panel (figure1b) is learned representation PC scatter (model activation space). Old RDM panel moved to supplementary S18.
- **Figure 3 bottom row is PC scatter** -- 4 panels showing representational geometry (Behavioral, CNN 8-class CLIP, AlexNet 1K, ViT-B/16 1K). RDMs moved to Figure 4 Panel A.
- **Figure 4 row 1 has RDMs** -- category-sorted RDMs (Behavioral, 8-class CLIP, 1000-class) with 8 super-category groupings. Row 2 has per-concept scatter + histogram. Dimension profiling exists as a standalone script but is not in the main figure.
- **Figure 5 has no schematic** -- directly shows data-efficiency line plots for NSD (early + ventral) and THINGS.
- **No DINO or ViT PCA in main figures** -- supplementary only. Main figures show AlexNet, CLIP, and Pixels PCA sources.
- **No NSD-Synthetic in main figures** -- supplementary only.
- **V4 (TVSD) in supplementary only** -- V1 and IT represent the extremes of the visual hierarchy.
