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

**Status:** Working draft — figures are organized into 5 main figures with subdirectories (`fig1/`–`fig5/`). Supplementary figures listed at the end.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Narrative Arc

The five main figures follow a progression: **method overview + representation analysis → neural data → behavioral data → per-concept analysis → data efficiency**.

1. **Figure 1** combines the method overview with representation analysis: the top row shows the PCA-based coarse-graining procedure via shared-PCA scatter (2-way, 4-way, 1000-way label coloring), and the bottom row shows that the *learned* representations are qualitatively different via PC scatter of model activations (2-way, 4-way, 1000-way side by side with image insets).
2. **Figure 2** presents neural alignment across species: macaque electrophysiology (TVSD) and human fMRI (NSD) side by side, with coarseness curves (raw Spearman ρ). Per-layer profiles in supplementary.
3. **Figure 3** presents behavioral alignment (THINGS): coarseness results, model comparison with pretrained baselines, and RDM visualizations explaining *why* coarse models win.
4. **Figure 4** digs into per-concept alignment: which semantic categories drive the coarse advantage, how broad is the effect, and which behavioral dimensions explain the pattern?
5. **Figure 5** introduces the data-efficiency paradigm: coarse vs fine-grained training at varying data scales.

---

## Figure 1: Method Overview + Representation Analysis

**Directory:** `manuscript/figures/fig1/` (top row), `manuscript/figures/fig2/` (bottom row)

**Narrative role:** Combines the method introduction with a visual demonstration that coarse-trained representations are qualitatively different. The top row shows how the coarse-graining procedure partitions images in a shared PCA space; the bottom row shows that the *learned* representations reorganize geometry in fundamentally different ways depending on label granularity.

### Layout — 2 rows × 3 columns

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Shared PCA:      │ Shared PCA:      │ Shared PCA:      │
│ 2-way coloring   │ 4-way coloring   │ 1000-way coloring│
│ (label space)    │ (label space)    │ (label space)    │
├──────────────────┼──────────────────┼──────────────────┤
│ Learned repr:    │ Learned repr:    │ Learned repr:    │
│ 2-way model      │ 4-way model      │ 1000-way model   │
│ (FC1 PCs +       │ (FC1 PCs +       │ (FC1 PCs +       │
│  image insets)   │  image insets)   │  image insets)   │
└──────────────────┴──────────────────┴──────────────────┘
```

### Top row (Panels A–C) — Shared PCA scatter (label space)

Three PC1 vs PC2 scatter plots of ImageNet images (1 per class, 1000 points), projected onto the same CLIP PCA axes. All three panels share identical coordinates — only the coloring changes:

- **A (2-way):** Median split on PC1 → 2 colors with decision boundary line.
- **B (4-way):** Median splits on PC1 then PC2 → 4 colors with decision boundary lines.
- **C (1000-way):** Each ImageNet class gets a unique color → continuous gradient, no structure.

- *Key message:* The coarse-graining procedure is simple and principled — median splits along PCA axes of a pretrained feature space. Same images, same coordinates, different label assignments.
- *Script:* `manuscript/figures/fig1/pc_scatter_explore.py`
- *Data:* `manuscript/figures/fig1/pc_scatter_shared_pca.npz`

### Bottom row (Panels D–F) — Learned representation scatter (model activation space)

Three side-by-side PC1 vs PC2 scatter plots of ImageNet activations (FC1 layer, L2-normalized), each from a model trained at a different granularity. Points colored by their respective coarse labels. Representative images shown as insets at class extremes and centroids.

- **D (2-way model):** Two broad clusters with rich internal variability.
- **E (4-way model):** Four well-separated clusters, each internally variable but categorically distinct.
- **F (1000-way model):** Smooth gradient — no clear class separation; coarse labels intermixed.

- *Key visual:* The bottom row mirrors the top row's 2-way / 4-way / 1000-way progression, but now shows how each training objective *reshapes* the learned geometry. Coarse models impose clear categorical boundaries; the 1000-way model distributes images along a continuous gradient.
- *Key message:* Coarse training fundamentally changes how the network organizes visual information — different geometry, not just reduced geometry. These representations cannot be recovered by dimensionality reduction of a fine-grained model.
- *Data source:* `experiments/representation_analysis/2pcs_compare/` (AlexNet-PCA labels, imagenet-mini-50, L2-normed FC1). Will need a `data_2way_alexnet.npz` in addition to existing `data_4way_alexnet.npz`.

### Observed results

**(A–C) Shared PCA scatter.** All 1000 points share the same (x, y) coordinates across panels. The 2-way panel shows a clean PC1 median split (natural vs man-made). The 4-way panel subdivides each half by PC2. The 1000-way panel shows no visible structure — a cloud of 1000 distinct colors.

**(D) 2-way model learned representations.** FC1 activations projected onto their top 2 PCs show two well-separated clusters with high variance explained by PC1. Image insets reveal the semantic split (e.g., natural scenes vs artifacts).

**(E) 4-way model learned representations.** Four well-separated clusters (PC1 ≈ 22.5%, PC2 ≈ 19.5% var.), each internally variable but categorically distinct. Image insets reveal the semantic content of each cluster.

**(F) 1000-way model learned representations.** Smooth gradient in PC space (PC1 ≈ 3.0%, PC2 ≈ 2.4% var.) — coarse labels are intermixed with no clear boundaries. The model distributes representations to separate 1000 classes, destroying the broad categorical structure.

**Cross-model RSA (described in text, supplementary).** Cross-model RSA (1000-way vs. coarse) increases from ρ ≈ 0.22 (2-way) to ρ ≈ 0.52 (64-way), but never approaches the inter-seed baseline (ρ ≈ 0.76). Projection control: projected-1K vs. coarse RSA is extremely low (ρ ≈ 0.03 at 2-class, ρ ≈ 0.35 at 64-class), confirming coarse features cannot be recovered by PCA of the fine-grained model.

---

## Figure 2: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig2/`

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

## Figure 3: Behavioral Alignment (THINGS)

**Directory:** `manuscript/figures/fig3/`

**Narrative role:** Present the behavioral alignment results — the most surprising finding. Coarse models *vastly* outperform 1000-way on human similarity judgments. This figure shows the result (coarseness log plot), compares against pretrained models, and visualizes *why* via RDMs. Per-concept analysis is in Figure 4; data efficiency is in Figure 5.

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

## Figure 4: Per-Concept Alignment Analysis

**Directory:** `manuscript/figures/fig4/`

**Narrative role:** Dig deeper into *which* concepts drive the coarse advantage on THINGS behavioral alignment. The scatter plot and histogram from the original Figure 3 are expanded here as standalone panels, with room for additional analyses.

### Layout — 1 row × 3 columns

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Per-concept         │ Per-concept           │ Dimension            │
│ scatter             │ advantage histogram   │ profiling            │
│ (CLIP 8-class       │                       │ (barh: top 25        │
│  vs. 1000-way)      │                       │  THINGS dims)        │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

### Panel A — Per-concept scatter plot (CLIP model)

Scatter plot of per-concept RSA contribution: **CLIP 8-class** (y-axis) vs. 1000-way (x-axis). Identity line for reference. Points color-coded by THINGS semantic category.

- *Key pattern:* ~70% of concepts fall above the diagonal (coarse model wins). Plants, animals, clothing accessories strongly favor coarse. Body parts, drinks favor 1000-way.
- *Key message:* The coarse model advantage is pervasive across most concept categories, not a niche effect.

### Panel B — Per-concept advantage histogram

Histogram of per-concept advantage: `(coarse_score - 1000way_score)` for each eval concept. Positive = coarse wins, negative = 1000-way wins.

- *Key visual:* Distribution clearly shifted right. Vertical line at zero divides coarse-advantage (green) from 1K-advantage (orange) bins.
- *Key message:* Quantifies that the advantage is broad — not driven by outlier concepts.

### Panel C — Semantic dimension profiling

Horizontal bar chart showing Spearman ρ between per-concept advantage (CLIP 8-class − 1000-way) and each of the 66 THINGS behavioral dimensions. Top 25 dimensions by |ρ| displayed. Bars sorted descending (most positive at top, most negative at bottom) with symmetric x-axis. Green = dimension loading favors 8-class model; red = favors 1000-class.

- *Key visual:* Animal-related, plant-related dimensions strongly favor coarse models; home/furnishing, metallic/artificial dimensions favor 1000-way.
- *Key message:* The coarse advantage is driven by high-level categorical dimensions (animate, natural), while fine-grained training excels on lower-level material/functional properties.
- *Script:* `manuscript/figures/fig4/dimension_profiling.py`

### Observed results

**(A) Per-concept scatter.** 1,207 of 1,854 concepts (~70%) fall above the diagonal. Green-colored clusters (plants, animals) are consistently above; orange clusters (body parts, tools) are below.

**(B) Histogram.** Δρ distribution is right-shifted with median ≈ +0.088. Spans approximately −0.50 to +0.75, with the positive tail substantially longer.

**(C) Dimension profiling.** Animal-related (ρ ≈ +0.19) and plant-related (ρ ≈ +0.17) dimensions most strongly favor coarse models. Home/furnishing (ρ ≈ −0.29) and metallic/artificial (ρ ≈ −0.21) most strongly favor 1000-way. All 66 dimensions significant after FDR correction.

---

## Figure 5: Data Efficiency

**Directory:** `manuscript/figures/fig5/`

**Narrative role:** Introduce a new analysis paradigm — varying the number of training images per class while holding granularity constant. Shows that coarse-trained models are more data-efficient than fine-grained models on behavioral alignment.

### Layout — 1 row × 3 columns

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Schematic           │ NSD                  │ THINGS               │
│ (paradigm)          │ (Ventral Stream)     │ (Behavioral)         │
│                     │ line plot            │ line plot            │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

### Panel A — Data-efficiency paradigm schematic

Schematic showing the experimental paradigm: same model architecture trained with coarse (8, 16, 32-class) vs fine (1000-class) labels, but varying the number of training images (5K, 10K, 50K, 1.2M).

- *Key message:* Introduces the data-efficiency question: does the coarse advantage persist when data is limited?

### Panel B — NSD Ventral Stream (line plot)

Line plot with 4 conditions (8, 16, 32, 1000-class) across 4 data scales (5K, 10K, 50K, 1.2M training images). Green shades for coarse models, orange for 1000-class.

- *Key visual:* Coarse models consistently outperform 1000-class at all data scales, with the gap narrowing at full scale.
- *Key message:* Coarse training provides a better inductive bias when data is limited, even for neural alignment.

### Panel C — THINGS Behavioral (line plot)

Same line-plot format as Panel B, showing behavioral alignment (Spearman ρ) across data scales.

- *Key visual:* The coarse advantage is dramatic and persistent — 1000-class never catches up, even at full ImageNet.
- *Key message:* Coarse training is not only better at full scale — it is vastly more data-efficient for behavioral alignment.

### Observed results

**(A)** Schematic (no data).

**(B) NSD Ventral Stream.** All coarse models (8, 16, 32-class) outperform 1000-class across all data scales. At 5K, coarse models achieve ρ ≈ 0.10 while 1000-class is at ρ ≈ 0.05. All models converge toward ρ ≈ 0.25 at full scale (1.2M), but coarse models maintain a slight edge.

**(C) THINGS Behavioral.** The headline data-efficiency result. At 5K, 8-class achieves ρ ≈ 0.45 while 1000-class reaches only ρ ≈ 0.27. The gap persists at all scales. At full ImageNet (1.2M), coarse models reach ρ ≈ 0.57 vs 1000-class at ρ ≈ 0.40.

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
| ~~**S13**~~ | *(moved to Figure 4C)* | | |
| **S14** | `supp_s14_image_collages.py` | Composite of example images where coarse wins vs. 1000-way wins | Visual intuition for which concepts benefit from coarse vs. fine-grained training. |
| **S15** | `supp_s15_pc_poles.py` | Most/least activating ImageNet images for top PCs of AlexNet + CLIP | PC1 separates natural/man-made; higher PCs capture progressively finer semantic axes. |
| **S16** | `supp_s16_levels.py` | Levels benchmark (Muttenthaler et al. 2025): 3×3 grid (metrics × triplet types) | Coarse models improve on between-class and class-boundary triplets; within-class converges at moderate granularity. |
| **S17** | `supp_s17_seed_variability.py` | Seed variability analysis | Score variability across seeds. |
| **S18** | `plot_class_rdms.py` | Class-level RDM grid (all granularity levels: 2,4,8,16,32,64,1000-way, CLIP-PCA) | Moved from old Figure 2A. Full progression of block-diagonal structure across all coarseness levels. |

---

## Directory Structure

```
manuscript/figures/
├── paper.md              # This file
├── fig_utils.py                # Shared constants, style, helpers
├── things_utils.py             # Shared THINGS plotting utilities
├── fig1/                       # Figure 1 top row: shared PCA scatter + method overview
│   ├── pc_scatter_explore.py   # Shared PCA scatter (2-way, 4-way, 1000-way coloring)
│   └── pc_scatter_shared_pca.png
├── fig2/                       # Figure 2: Neural alignment (TVSD + NSD)
│   ├── figure2.py              # Combined TVSD + NSD figure
│   └── figure2.png
├── fig3/                       # Figure 3: THINGS behavioral — headline result + RDMs
│   ├── figure3.py              # Schematic + coarseness + model comparison + RDMs
│   └── figure3.png
├── fig4/                       # Figure 4: Per-concept alignment analysis
│   ├── figure4.py              # Scatter + histogram
│   ├── dimension_profiling.py  # Semantic dimension profiling (Panel C)
│   └── figure4.png
├── fig5/                       # Figure 5: Data efficiency
│   ├── figure5.py              # Data-efficiency line plots
│   └── figure5.png
└── supplementary/              # 18 supplementary figures (S1–S18)
    ├── README.md               # Index, run commands, data sources
    ├── supp_s1_training_summary.py
    ├── supp_s2_summary_bars.py
    ├── ...
    └── plot_class_rdms.py
```

---

## Design Notes

- **Consistent color scheme across Figures 2 & 3:**
  - PCA source models: AlexNet (medium blue `#6baed6`, circle), CLIP (dark blue `#08519c`, square), Pixels (brown `#8c564b`, triangle-down). ViT moved to supplementary.
  - 1000-way baseline: Warm amber (`#e8963e`, diamond)
  - Untrained baseline: Light gray dashed line (`#AAAAAA`)
  - Supplementary figures retain the original 4-architecture palette (AlexNet teal, CLIP purple, ViT crimson, Pixels brown) in `fig_utils.py`.
- **Raw Spearman ρ (Figures 2 & 3):** All coarseness plots show raw Spearman ρ values.
- **Per-layer profiles moved to supplementary (S3).** Main figures show only coarseness curves.
- **Reconstruction controls in supplementary (S4).** Neural reconstruction for TVSD/NSD; THINGS reconstruction also in supplementary.
- **Summary bars in supplementary (S2).** Cross-dataset pretrained vs. scratch comparison.
- **Figure 1 combines method + representation analysis** — top row is shared PCA scatter (label space), bottom row is learned representation PC scatter (model activation space). Old RDM panel moved to supplementary S18.
- **Log-scale x-axis** for all coarseness plots (Figures 2 and 3).
- **Schematics should be simple** — the reader should grasp each dataset in 5–10 seconds.
- **Reconstruction controls in supplementary** — not in main figures. This keeps Figures 2–4 focused on results.
- **Figure 3 is the climax** — the behavioral result is the most surprising, shown via coarseness + model comparison + RDMs.
- **Figure 4 is the deep dive** — per-concept analysis showing which semantic categories drive the coarse advantage.
- **Figure 5 is the practical implication** — data efficiency shows coarse training is not only better but also more sample-efficient.
- **Class-level RDMs in supplementary (S18)** — moved from old Figure 2A. Supplementary version includes all granularity levels (2, 4, 8, 16, 32, 64, 1000-way), not just alternating.
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
