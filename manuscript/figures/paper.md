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

**Status:** Working draft — figures are organized into 6 main figures with subdirectories (`fig1/`–`fig6/`). Supplementary figures listed at the end.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Narrative Arc

The six main figures follow a progression: **method overview → representation analysis → neural data → behavioral data → per-concept analysis → data efficiency**.

1. **Figure 1** is a schematic overview: the PCA-based coarse-graining procedure, the DNN training paradigm, how alignment is measured (RSA), and the evaluation domains (brain and behavior).
2. **Figure 2** establishes that coarse-trained representations are genuinely different from 1000-way representations — via class-level RDMs, cross-model RSA, projection controls, and PC-space visualizations with image insets showing the geometric reorganization.
3. **Figure 3** presents neural alignment across species: macaque electrophysiology (TVSD) and human fMRI (NSD) side by side, with coarseness curves (raw Spearman ρ). Per-layer profiles in supplementary.
4. **Figure 4** presents behavioral alignment (THINGS): coarseness results, model comparison with pretrained baselines, and RDM visualizations explaining *why* coarse models win.
5. **Figure 5** digs into per-concept alignment: which semantic categories drive the coarse advantage, and how broad is the effect?
6. **Figure 6** introduces the data-efficiency paradigm: coarse vs fine-grained training at varying data scales.

---

## Figure 1: Method Overview (Schematic)

**Directory:** `manuscript/figures/fig1/`

**Narrative role:** A purely schematic figure that introduces the approach. No data, no results — just the method pipeline and evaluation framework. This is the "how we did it" figure.

**Status:** Schematic — to be composed manually (e.g., in Illustrator/Inkscape). Code-generated components (PC scatter) have been moved to Figure 2C.

### Panel A — The coarse-graining procedure

Schematic of the pipeline: ImageNet images → pretrained source model (AlexNet, CLIP, ViT, or raw Pixels) → PCA on feature space (or pixel space) → recursive median splits along top PCs → coarse label sets (2, 4, 8, 16, 32, 64 classes).

Follow the spirit of the NeurIPS schematic (`experiments/neurips_2025/fig1/schematic_imagenet_pca.png`) — a clean, linear pipeline diagram. Show the flow from images to features to PCA projection to class assignments. Include example images at each split to give intuition for what PC1, PC2, etc. separate (e.g., PC1 roughly separates man-made vs. natural).

- *Key message:* The procedure is principled (data-driven, not arbitrary taxonomy), produces balanced classes, and uses only visual statistics — no linguistic or semantic labels.

### Panel B — DNN training schematic

Minimal diagram showing the training paradigm: same architecture (CustomCNN / AlexNet-style) × same ~1.26M ImageNet images × same training protocol (SGD, 20 epochs, same augmentations), only varying the output classification head (2, 4, 8, 16, 32, 64, or 1000 classes). Three seeds per condition.

- *Key message:* The **only** variable is label granularity. Architecture, data, and training procedure are identical across all conditions.

### Panel C — RSA methodology schematic

Schematic showing how representational similarity analysis is computed: model activations → RDM (representational dissimilarity matrix) → comparison with neural/behavioral RDMs via Spearman correlation.

- *Key message:* RSA provides a common currency to compare model representations with brain and behavioral data.

### Panel D — Brain and behavior evaluation domains

Schematic showing the three evaluation benchmarks: (1) macaque electrophysiology (TVSD — V1, V4, IT), (2) human fMRI (NSD — early and ventral visual stream), (3) human behavioral similarity judgments (THINGS triplet task).

- *Key message:* We test across species (macaque + human), measurement modalities (spiking + BOLD), and levels (neural + behavioral).

---

## Figure 2: Coarse Representations Are Fundamentally Different

**Directory:** `manuscript/figures/fig2/`

**Narrative role:** Establish that the internal representations learned from coarse supervision are *qualitatively distinct* from 1000-way representations. The RDMs show how internal structure changes with granularity, and the PC scatter with image insets gives immediate visual intuition for geometric reorganization. Cross-model RSA and projection control results are described in text and available in supplementary.

### Panel A — 2×3 grid of class-level RDMs

Six rank-normalized class-level RDMs (1000×1000, Pearson dissimilarity, FC1 layer) arranged in a 2×3 grid, sorted by 8 WordNet super-categories with colored sidebars. The grid shows 4-way, 8-way, 16-way, 32-way, 64-way, and 1000-way models (CLIP-PCA labels), progressing from coarsest to finest.

- *Key visual:* The block-diagonal structure (within-category similarity) progressively sharpens from 4-way to 1000-way, but even the coarsest models show meaningful category organization.
- *Key message:* Coarse training produces qualitatively distinct internal geometry — not simply a blurred version of fine-grained representations.

### Panel B — PC1/PC2 scatter with image insets

Two vertically stacked PC1 vs. PC2 scatter plots of ImageNet activations (FC1 layer, L2-normalized), colored by 4-way AlexNet-PCA labels:

**Top:** Fine-grained (1000-way) model — smooth gradient in PC space, no clear class separation.
**Bottom:** Coarsened (4-way) model — distinct clusters with rich internal variability.

A subset of representative points (~3 per class) show actual ImageNet thumbnails as insets, with borders colored by class. Inset points are selected at class extremes and centroids for maximum informativeness.

- *Key visual:* The 1000-way model distributes images along a continuous gradient; the coarse model reorganizes the same images into clearly separated groups. The image insets make this concrete — you can *see* that each cluster contains visually coherent images.
- *Key message:* Coarse training fundamentally changes how the network organizes visual information — different geometry, not just reduced geometry.
- *Data source:* `experiments/representation_analysis/2pcs_compare/data_4way_alexnet.npz` (AlexNet-PCA labels, imagenet-mini-50, L2-normed FC1).

### Observed results

**(A) Class-level RDM grid.** Six RDMs showing the progression from 4-way to 1000-way. The 1000-way RDM shows fine-grained within-category structure with sharp diagonal blocks. Coarser models show progressively broader block structure — the 4-way model has large uniform blocks while the 64-way model approaches the 1000-way pattern but with coarser boundaries.

**(B) PC scatter with image insets (FC1, AlexNet-PCA 4-way).** Top panel: the 1000-way model's FC1 activations projected onto their own top 2 PCs show a smooth gradient (PC1 ≈ 3.0%, PC2 ≈ 2.4% var.) — 4-way PCA labels are intermixed with no clear boundaries. Bottom panel: the 4-way model's FC1 activations show four well-separated clusters (PC1 ≈ 22.5%, PC2 ≈ 19.5% var.), each internally variable but categorically distinct. Image insets reveal the semantic content of each cluster. The coarse model imposes a fundamentally different geometry, not just a lower-dimensional version of the fine-grained geometry.

**Cross-model RSA (described in text, supplementary).** Cross-model RSA (1000-way vs. coarse) increases from ρ ≈ 0.22 (2-way) to ρ ≈ 0.52 (64-way), but never approaches the inter-seed baseline (ρ ≈ 0.76). Projection control: projected-1K vs. coarse RSA is extremely low (ρ ≈ 0.03 at 2-class, ρ ≈ 0.35 at 64-class), confirming coarse features cannot be recovered by PCA of the fine-grained model.

---

## Figure 3: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present neural alignment results from both macaque electrophysiology (TVSD) and human fMRI (NSD) in a single unified figure. This cross-species layout immediately demonstrates that the coarseness finding is robust — it holds in spiking data and BOLD fMRI, across early and late visual regions, and across two completely different species. **All coarseness plots are normalized to percentage of the 1000-way baseline** (1000-way = 100%), making the central claim visually immediate. No reconstruction controls here — those are in Supplementary S4.

### Layout — 2 rows × 2 columns

Each dataset occupies **one row**. Each row contains 2 coarseness panels (early + late region). Per-layer profiles moved to supplementary (S3). ViT PCA labels moved to supplementary (alongside DINOv3).

```
┌──────────────┬──────────────┐
│ TVSD V1      │ TVSD IT      │
│ Coarseness   │ Coarseness   │
├──────────────┼──────────────┤
│ NSD Early    │ NSD Ventral  │
│ Visual       │ Visual       │
│ Coarseness   │ Coarseness   │
└──────────────┴──────────────┘
```

**2 rows × 2 columns = 4 panels total.** No in-figure schematics — dataset details are described in the figure caption.

### Coarseness plots (4 panels — raw Spearman ρ)

Y-axis shows raw Spearman ρ. Log₂ x-axis (2 → 1000). Three PCA source models (AlexNet, CLIP, Pixels) as separate markers/colors, plus 1000-way baseline (diamond) and untrained baseline (dashed line).

**Color scheme:** AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (brown `#8c564b`, triangle-down), 1K (warm amber `#e8963e`, diamond). ViT moved to supplementary. Same color scheme used in Figure 4B.

**Panels:**
1. **TVSD V1 (early):** Coarse models match or exceed 1000-way. Flat across granularity.
2. **TVSD IT (late):** Gradual increase, 32–64 classes approach 1000-way.
3. **NSD Early Visual Stream:** Similar to TVSD V1 — saturation at low granularity.
4. **NSD Ventral Visual Stream:** Gradual increase, approaching 1000-way by 32–64 classes.

- *Error bars — TVSD:* ±1.96 SEM across 2 monkeys × 3 seeds.
- *Error bars — NSD:* Bootstrap CIs aggregated across 8 subjects × 3 seeds.

### Panel labels

Panel labels A–D across the 2×2 grid:
- A: TVSD V1, B: TVSD IT, C: NSD Early Visual, D: NSD Ventral Visual

### Per-layer profiles (moved to supplementary S3)

Full per-layer RSA profiles (all 7 granularity levels, 14 layer taps) available in Supplementary S3.

### Observed results

**(A) TVSD V1 coarseness.** AlexNet and CLIP cluster at ρ ≈ 0.16–0.18 across all granularity levels (2–64), matching or exceeding the 1000-way baseline (ρ ≈ 0.15). Pixels is lower (ρ ≈ 0.10–0.15). Flat curve — even 2-class models match 1000-way for V1.

**(B) TVSD IT coarseness.** Clear monotonic ramp: 2-class at ρ ≈ 0.12, increasing to ρ ≈ 0.17–0.19 by 64 classes, approaching 1000-way (ρ ≈ 0.16). Pixels lags substantially. Untrained well separated at ρ ≈ 0.05.

**(C) NSD Early Visual Stream coarseness.** AlexNet and CLIP at ρ ≈ 0.19–0.21 across all levels. 1000-way at ρ ≈ 0.20. Pixels lower at ρ ≈ 0.10. Untrained at ρ ≈ 0.12.

**(D) NSD Ventral Visual Stream coarseness.** Gradual ramp: 2-class at ρ ≈ 0.15, rising to ρ ≈ 0.25 by 64 classes. 1000-way at ρ ≈ 0.21. CLIP converges fastest. Untrained well separated at ρ ≈ 0.04.

---

## Figure 4: Behavioral Alignment (THINGS)

**Directory:** `manuscript/figures/fig4/`

**Narrative role:** Present the behavioral alignment results — the most surprising finding. Coarse models *vastly* outperform 1000-way on human similarity judgments. This figure shows the result (coarseness log plot), compares against pretrained models, and visualizes *why* via RDMs. Per-concept analysis is in Figure 5; data efficiency is in Figure 6.

### Layout — 2 rows

```
┌───────────────┬──────────────────┬──────────────────────────────────┐
│ Schematic     │ Coarseness       │ Model Comparison                 │
│ (THINGS)      │ (raw Spearman ρ) │ (coarse vs 1K bars +             │
│               │                  │  pretrained scatter)             │
├───────────────┴──────────────────┴──────────────────────────────────┤
│                     3 RDMs side by side                             │
│  [Human behavioral]    [Coarse (CLIP 8-way)]    [1000-way model]   │
└────────────────────────────────────────────────────────────────────┘
```

### Panel A — THINGS schematic

Schematic of the THINGS behavioral similarity task: triplet odd-one-out judgments, how behavioral RDMs are constructed, and the comparison with model RDMs.

- *Key message:* Introduces the behavioral benchmark before showing results.

### Panel B — Coarseness log plot (raw Spearman ρ)

Y-axis shows raw Spearman ρ. Log₂ x-axis (2 → 1000). Three PCA architectures (AlexNet, CLIP, Pixels — same blue/amber color scheme as Figure 3). ViT moved to supplementary.

- *Key pattern:* All coarse models sit well **above** the 1000-way baseline — the coarse advantage is dramatic. Even 2-class models exceed 1000-way. This is the headline result.
- *Error bars:* Bootstrap 95% CIs across 3 seeds.
- *Architectures:* AlexNet (medium blue circle), CLIP (dark blue square), Pixels (brown triangle-down), 1K (warm amber diamond).

### Panel C — Model comparison (coarse vs 1K + pretrained)

Two NeurIPS-style bars: best coarse model (dark blue) vs 1000-way (amber), plus a grouped scatter of pretrained models (supervised, self-supervised, vision-language) with architecture markers (CNN pentagon, ViT star). A dashed reference line from the coarse bar extends into the pretrained region.

- *Key visual:* The coarse-trained model (trained from scratch) matches or exceeds many large pretrained models on behavioral alignment.
- *Key message:* Coarse supervision is not just better than 1000-way — it competes with the best pretrained vision models.

### Panel D — Category-annotated RDMs

Three RDMs side by side — the most visually striking evidence for *why* coarse models win:

1. **Human behavioral RDM** — ground truth similarity structure from THINGS triplet judgments
2. **Coarse model RDM** (CLIP 8-class) — captures the broad block structure
3. **1000-way model RDM** — imposes finer distinctions that don't match human judgments

Concepts sorted by the 27 THINGS semantic categories with boundary lines overlaid.

- *Key visual:* The coarse model RDM captures the broad categorical block structure of human similarity much better than the 1000-way model, which over-differentiates within categories.
- *Key message:* Fine-grained training emphasizes within-category distinctions at the expense of the broad between-category structure that dominates human similarity judgments.

### Observed results

**(A)** Schematic (no data).

**(B) Coarseness log plot.** The headline result of the paper. All coarse models sit well above the 1000-way baseline. CLIP labels (dark blue) are strongest: ρ ≈ 0.55–0.57. AlexNet (medium blue): ρ ≈ 0.44–0.48. Pixels (brown) starts low at ρ ≈ 0.10 (2-class), rises to ρ ≈ 0.23 at 64-class but never reaches 1000-way (ρ ≈ 0.39). The untrained baseline is at ρ ≈ 0.20.

**(C) Model comparison.** Best coarse bar (ρ ≈ 0.57) substantially exceeds 1000-way bar (ρ ≈ 0.39). Among pretrained models, CLIP-L/14 and DINOv2 approach but do not exceed the coarse-trained model. Supervised CNNs cluster well below the coarse reference line.

**(D) Category-annotated RDMs.** Three 1,854 × 1,854 RDMs sorted by 27 semantic categories. The human behavioral RDM (ρ = 0.538) shows clear block-diagonal structure. The CLIP 8-class RDM (ρ = 0.538) captures this block structure remarkably well. The 1000-way RDM (ρ = 0.392) has weaker block boundaries — it over-differentiates within categories.

---

## Figure 5: Per-Concept Alignment Analysis

**Directory:** `manuscript/figures/fig5/`

**Narrative role:** Dig deeper into *which* concepts drive the coarse advantage on THINGS behavioral alignment. The scatter plot and histogram from the original Figure 4 are expanded here as standalone panels, with room for additional analyses.

### Layout — 1 row × 2 columns

```
┌─────────────────────┬──────────────────────┐
│ Per-concept         │ Per-concept           │
│ scatter             │ advantage histogram   │
│ (CLIP 8-class       │                       │
│  vs. 1000-way)      │                       │
└─────────────────────┴──────────────────────┘
```

### Panel A — Per-concept scatter plot (CLIP model)

Scatter plot of per-concept RSA contribution: **CLIP 8-class** (y-axis) vs. 1000-way (x-axis). Identity line for reference. Points color-coded by THINGS semantic category.

- *Key pattern:* ~70% of concepts fall above the diagonal (coarse model wins). Plants, animals, clothing accessories strongly favor coarse. Body parts, drinks favor 1000-way.
- *Key message:* The coarse model advantage is pervasive across most concept categories, not a niche effect.

### Panel B — Per-concept advantage histogram

Histogram of per-concept advantage: `(coarse_score - 1000way_score)` for each eval concept. Positive = coarse wins, negative = 1000-way wins.

- *Key visual:* Distribution clearly shifted right. Vertical line at zero divides coarse-advantage (green) from 1K-advantage (orange) bins.
- *Key message:* Quantifies that the advantage is broad — not driven by outlier concepts.

### Observed results

**(A) Per-concept scatter.** 1,207 of 1,854 concepts (~70%) fall above the diagonal. Green-colored clusters (plants, animals) are consistently above; orange clusters (body parts, tools) are below.

**(B) Histogram.** Δρ distribution is right-shifted with median ≈ +0.088. Spans approximately −0.50 to +0.75, with the positive tail substantially longer.

---

## Figure 6: Data Efficiency

**Directory:** `manuscript/figures/fig6/`

**Narrative role:** Introduce a new analysis paradigm — varying the number of training images per class while holding granularity constant. Shows that coarse-trained models are more data-efficient than fine-grained models on behavioral alignment.

### Layout — 1 row × 2 columns

```
┌─────────────────────┬──────────────────────┐
│ Schematic           │ Data-efficiency       │
│ (paradigm)          │ bars                  │
│                     │ (8-class vs 1000-     │
│                     │  class at 4 scales)   │
└─────────────────────┴──────────────────────┘
```

### Panel A — Data-efficiency paradigm schematic

Schematic showing the experimental paradigm: same model architecture trained with coarse (8-class) vs fine (1000-class) labels, but varying the number of training images per class (5, 10, 50, ~1300).

- *Key message:* Introduces the data-efficiency question: does the coarse advantage persist when data is limited?

### Panel B — Data-efficiency paired bars

Paired bars at 4 data scales: 5, 10, 50, ~1300 images per class. Each pair compares coarse 8-class (dark blue) vs fine 1000-class (amber) on THINGS behavioral alignment.

- *Key visual:* The coarse advantage is present at all data scales and grows larger with less data.
- *Key message:* Coarse training is not only better at full scale — it is also more data-efficient.

### Observed results

**(A)** Schematic (no data).

**(B) Data-efficiency bars.** At all 4 data scales, coarse 8-class exceeds 1000-class. The gap is largest at small scales (5–10 images/class) where the 1000-way model struggles but the coarse model maintains strong alignment.

---

## Supplementary Figures

All scripts live in `manuscript/figures/supplementary/` and are run from the project root. Full details in `manuscript/figures/supplementary/README.md`. For detailed **message** and **result** descriptions of each figure, see `manuscript/figures/supplementary/figure_descriptions.md`.

### Theme 1: Training Validation

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S1** | `supp_s1_training_summary.py` | Test accuracy vs. number of classes (log scale), AlexNet-PCA labels, 3 seeds | All models converge; accuracy monotonically decreases from ~96% (2-class) to ~74% (1000-way). No training failure. |

### Theme 2: Extended Main Results

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S2** | `supp_s2_summary_bars.py` | 5-condition bars (ViT, CLIP, Untrained, 1K, Best Coarse) across all benchmarks | Cross-dataset pretrained vs. trained-from-scratch comparison; coarse matches or exceeds 1000-way everywhere. |
| **S3** | `supp_s3_full_per_layer.py` | Per-layer RSA for all 7 granularity levels (2–1000), 2×3 grid | Complete per-layer profiles; early regions overlap, late regions fan out by granularity. |
| **S4** | `supp_s4_neural_reconstruction.py` | Alignment vs. PCs retained (top-k) for TVSD V1/V4/IT and NSD early/ventral | Reconstruction control: alignment plateaus by k ≈ 10–20 PCs; not a dimensionality artifact. |
| **S5** | `supp_s5_things_architectures.py` | Individual THINGS normalized coarseness curves per PCA source (1×4 grid) | AlexNet/CLIP/ViT all exceed 100%; Pixels reaches ~55–60% (weakest but still informative). |

### Theme 3: Anatomical Detail

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S6** | `supp_s6_finegrained_roi.py` | Normalized coarseness for 6 individual ROIs (V1, V2, V3, hV4, FFA, PPA), 2×3 grid, all 4 architectures | Coarseness effect at fine anatomical resolution; FFA/PPA show steepest ramp toward 1000-way. |

### Theme 4: Robustness & Generalization

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S7** | `supp_s7_nsd_synthetic.py` | Normalized coarseness on NSD-Synthetic (220 OOD stimuli), AlexNet + CLIP | Coarse alignment holds on synthetic stimuli (noise patterns, gratings, Mooney images). |
| **S8** | `supp_s8_stimulus_robustness.py` | RSA under stimulus subsampling (10–100%, 50 reps) | Alignment estimates are stable under subsampling; coarse >= fine pattern robust to stimulus set size. |
| **S9** | `supp_s9_score_distributions.py` | Violin/strip plots of score distributions across subjects × seeds, 2×3 grid | Full spread of individual scores; effects are not driven by averaging artifacts. |

### Theme 5: Alternative Labels

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S10** | `supp_s10_dinov2.py` | Normalized coarseness curves for ViT + DINOv3 labels, 2×3 grid (all 6 dataset–region combos) | Main findings replicate with ViT and DINOv3 — not contingent on any single PCA source geometry. |
| **S11** | `supp_s11_wordnet.py` | WordNet hierarchy-derived coarse labels across all benchmarks | PCA-based and taxonomy-based labels both show the coarseness effect; not an artifact of label source. |

### Theme 6: Representational & Perceptual Analysis

| Figure | Script | What it shows | Key takeaway |
|--------|--------|---------------|-------------|
| **S12** | `supp_s12_representation_summary.py` | Eigenspectrum, participation ratio, TwoNN intrinsic dim, sparsity across granularity (2×2) | Coarse models concentrate variance in fewer dimensions with higher sparsity. |
| **S13** | `supp_s13_dimension_profiling.py` | Horizontal bar chart: top 25 THINGS behavioral dimensions correlated with per-concept coarse advantage | Animal/plant dimensions drive coarse advantage; home/furnishing dimensions favor 1000-way. |
| **S14** | `supp_s14_image_collages.py` | Composite of example images where coarse wins vs. 1000-way wins | Visual intuition for which concepts benefit from coarse vs. fine-grained training. |
| **S15** | `supp_s15_pc_poles.py` | Most/least activating ImageNet images for top PCs of AlexNet + CLIP | PC1 separates natural/man-made; higher PCs capture progressively finer semantic axes. |
| **S16** | `supp_s16_levels.py` | Levels benchmark (Muttenthaler et al. 2025): 3×3 grid (metrics × triplet types) | Coarse models improve on between-class and class-boundary triplets; within-class converges at moderate granularity. |

---

## Directory Structure

```
manuscript/figures/
├── paper.md              # This file
├── fig_utils.py                # Shared constants, style, helpers
├── things_utils.py             # Shared THINGS plotting utilities
├── fig1/                       # Method overview (schematic only)
│   └── (schematics — coarse-graining, training, RSA, evaluation domains)
├── fig2/                       # Representations are different
│   ├── figure2.py              # RDMs + PC scatter w/ image insets
│   └── figure2.png
├── fig3/                       # Combined neural data (TVSD + NSD)
│   ├── figure3.py              # Combined TVSD + NSD figure
│   └── figure3.png
├── fig4/                       # THINGS behavioral — headline result + RDMs
│   ├── figure4.py              # Schematic + coarseness + model comparison + RDMs
│   └── figure4.png
├── fig5/                       # Per-concept alignment analysis
│   ├── figure5.py              # Scatter + histogram
│   └── figure5.png
├── fig6/                       # Data efficiency
│   ├── figure6.py              # Schematic + data-efficiency bars
│   └── figure6.png
└── supplementary/              # 16 supplementary figures (S1–S16)
    ├── README.md               # Index, run commands, data sources
    ├── supp_s1_training_summary.py
    ├── supp_s2_summary_bars.py
    ├── ...
    └── supp_s16_levels.py
```

---

## Design Notes

- **Consistent color scheme across Figures 3 & 4:**
  - PCA source models: AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (brown `#8c564b`, triangle-down). ViT moved to supplementary.
  - 1000-way baseline: Warm amber (`#e8963e`, diamond)
  - Untrained baseline: Light gray dashed line (`#AAAAAA`)
  - Supplementary figures retain the original 4-architecture palette (AlexNet teal, CLIP purple, ViT crimson, Pixels brown) in `fig_utils.py`.
- **Raw Spearman ρ (Figures 3 & 4):** All coarseness plots show raw Spearman ρ values.
- **Per-layer profiles moved to supplementary (S3).** Main figures show only coarseness curves.
- **Reconstruction controls in supplementary (S4).** Neural reconstruction for TVSD/NSD; THINGS reconstruction also in supplementary.
- **Summary bars in supplementary (S2).** Cross-dataset pretrained vs. scratch comparison.
- **Figure 1 is schematic-only** — no code-generated panels. PC scatter moved to Figure 2C. Expectations panel removed.
- **Figure 2 Panel C** — PC scatter with image insets (from `experiments/representation_analysis/2pcs_compare/data_4way.npz`). Moved from former Figure 1C.
- **Log-scale x-axis** for all coarseness plots (Figures 3 and 4).
- **Schematics should be simple** — the reader should grasp each dataset in 5–10 seconds.
- **Reconstruction controls in supplementary** — not in main figures. This keeps Figures 3–5 focused on results.
- **Figure 4 is the climax** — the behavioral result is the most surprising, shown via coarseness + model comparison + RDMs.
- **Figure 5 is the deep dive** — per-concept analysis showing which semantic categories drive the coarse advantage.
- **Figure 6 is the practical implication** — data efficiency shows coarse training is not only better but also more sample-efficient.
- **No DINO or ViT PCA in main figures** — supplementary only. Main figures show AlexNet, CLIP, and Pixels PCA sources.
- **No NSD-Synthetic in main figures** — supplementary only.
- **V4 (TVSD) in supplementary only** — V1 and IT represent the extremes of the visual hierarchy.
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
