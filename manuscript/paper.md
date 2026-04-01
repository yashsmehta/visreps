# Carving nature at its joints: a coarse feedback signal for learning human-aligned visual representations

---

## Abstract

The AI field has progressively moved toward richer training objectives — from supervised classification over a thousand categories to self-supervised methods that treat each image as its own class. In parallel, computational neuroscience has found that these increasingly powerful models also tend to produce representations that better match the brain, reinforcing an untested assumption: that rich learning signals are necessary for brain-like representations. Here we directly test this assumption by asking how coarse a training signal can be while still aligning with biological vision. Using a data-driven method that partitions images into broad categories (2, 4, 8, … 64) without manual annotation, we train hundreds of neural networks on ImageNet — holding architecture, dataset, and all hyperparameters constant — and evaluate them against human behavioral similarity judgments, macaque single-neuron recordings, and human fMRI responses. We find that networks trained to distinguish as few as eight broad categories develop representations more aligned with human perception than any fine-grained supervised or large-scale pretrained model we tested, a result that holds across every architecture evaluated. This alignment extends throughout the visual hierarchy and persists when training data is reduced to 1% of the full dataset. Our results show that alignment with biological vision depends not on how much information a learning signal carries, but on whether it partitions the input space along perceptually natural boundaries — suggesting that the computational principles underlying human-like visual representations may be far simpler than previously assumed.

---

**Status:** All updated figures from Fig. 1 to Fig. 6 are complete. Supplementary figures are maintained separately in `manuscript/figures/supplementary/`.

**PCA source models in main figures:** AlexNet, CLIP, Pixels. ViT and DINOv3-derived labels in supplementary only.

---

## Paper at a Glance

**Core question:** How coarse can the training signal be while still producing representations aligned with human perception and neural processing?

**Method:** Recursive PCA median splits on pretrained features → data-driven coarse labels (2, 4, 8, 16, 32, 64 classes). Train hundreds of CNNs on ImageNet with these labels, holding everything else constant.

**Key findings:**
1. **Coarse ≠ less** — Coarse-trained networks learn qualitatively different representations, not just reduced versions of fine-grained ones (Fig. 2)
2. **Neural alignment is preserved** — Models trained on as few as 8–32 classes match 1000-way alignment with both macaque V1/IT and human early/ventral visual cortex (Fig. 3)
3. **Behavioral alignment is *improved*** — Coarse models substantially *exceed* 1000-way on human similarity judgments (THINGS), and compete with the best large-scale pretrained models (Fig. 4)
4. **The advantage is pervasive** — ~82% of individual concepts favor the coarse model; the benefit spans all semantic categories (Fig. 5)
5. **Generalizes across architectures** — The pattern holds for ResNet-50, ConvNeXt, and ViT-B/16 (Fig. 6)
6. **Data efficient** — Coarse models on 1% of ImageNet outperform 1000-way on the full dataset (Fig. 4D)

**Paper structure:**
- **Introduction** — Field assumes richer signals → better alignment; we test the opposite
- **Fig. 1** — Schematic: coarse-graining method + experimental pipeline
- **Fig. 2** — Representation analysis: coarse training reshapes geometry fundamentally
- **Fig. 3** — Neural alignment: cross-species (macaque TVSD + human NSD)
- **Fig. 4** — Behavioral alignment: THINGS (headline result) + pretrained comparison + data efficiency
- **Fig. 5** — Per-concept analysis: RDMs + scatter + histogram showing broad advantage
- **Fig. 6** — Architecture generalization: ResNet-50, ConvNeXt, ViT-B/16
- **Discussion** — Coarse signals suffice because they carve along perceptually relevant boundaries

---

## Paper Narrative

### Introduction

DNNs trained on fine-grained classification (e.g., 1,000-category ImageNet) yield representations surprisingly well-aligned with primate visual cortex. This catalyzed the neuroAI field toward increasingly rich training signals — from hundreds of supervised categories to self-supervised and single-image objectives. The prevailing assumption: richness in the learning signal drives brain-like representations.

Yet this sits in tension with biological development — children initially carve the visual world into broad, coarse categories, only gradually acquiring finer distinctions. **Central question:** how coarse can the supervisory signal be while still producing representations that align with human perception and neural processing?

We explore the opposite direction from the field's trajectory. We developed a data-driven coarse-graining method (recursive PCA median splits → 2, 4, 8, 16, … classes) and trained hundreds of networks on ImageNet with these coarse signals — holding dataset, architecture, and all hyperparameters constant. This isolates the effect of supervisory signal granularity, evaluated against human behavioral judgments, macaque single-neuron recordings, and human fMRI.

### Section: Coarse-graining the input space (Fig. 1)

Neural representations in visual cortex are organized along a few dominant axes of variation (animacy, real-world size, etc.). We reasoned that principal axes of a learned visual representation could serve as a natural basis for partitioning images at multiple scales. Extract latent representations from a pretrained model (CLIP), perform PCA. Median split along PC1 → 2 classes (recovers animate–inanimate without manual labeling). Split each group along PC2 → 4 classes. Continue recursively → binary tree of 8, 16, 32, … classes. Not dependent on choice of pretrained model — qualitatively identical splits from AlexNet, ViT, DINOv3 (Supplementary).

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

## Figures

The six main figures follow a progression: **schematic → representation analysis → neural data → behavioral data → per-concept analysis → architecture generalization**.

---

### Figure 1: Experimental Overview

**Directory:** `manuscript/figures/fig1/` · **Script:** `plot_label_space.py`

#### Figure caption

**Fig. 1 | Coarse-graining the supervisory signal and measuring biological alignment.**
**a**, Data-driven coarse-graining of ImageNet. Left: the 1,000 ImageNet categories shown in the principal component space of image representations (one point per class). Right: a median split along PC1 divides all images into 2 broad classes; splitting each group again along PC2 yields 4 classes. Continuing this recursive halving produces 8, 16, 32 and 64 classes, each level doubling the previous, without any manual annotation. The only input to this procedure is a set of image representations — we experiment across a variety of sources, from a simple AlexNet to vision–language models such as CLIP, as well as raw pixel features, demonstrating that the approach does not depend on any particular pretrained system (Extended Data). Separate networks are then trained from scratch at each granularity level (2–1,000 classes), with architecture, dataset and all other hyperparameters held constant, isolating the effect of the supervisory signal.
**b**, Measuring alignment via representational similarity analysis (RSA). For a shared set of stimuli, pairwise similarity matrices are computed from model activations (left) and from biological measurements (right). Alignment is the correlation between these matrices, providing a common currency for comparison across three scales of measurement: single-neuron spiking responses in macaque visual cortex, human behavioural similarity judgements and whole-brain functional magnetic resonance imaging in humans.

---

### Figure 2: Categorical Nature of Representations

**Directory:** `manuscript/figures/fig2/` · **Script:** `figure2.py` → `plot_representations.py`

#### Figure caption

**Fig. 2 | Coarse supervision learns more categorical representations.**
**a**,**b**, PCA projections (PC1 versus PC2) of penultimate-layer activations from two convolutional neural networks (CNNs) trained on identical ImageNet images but with different supervisory signals. Each panel displays a mosaic of image thumbnails positioned at their coordinates in activation space; points are coloured by the four-way coarse labels derived from recursive PCA median splits of pretrained AlexNet representations (Fig. 1). Insets (top left) show dot-scatter views of the full training-set distribution (*n* = 50,000). **a**, A CNN trained with 1,000 ImageNet classes produces a smooth, diffuse representation in which the four coarse categories intermix with no clear boundaries. **b**, A CNN trained with only four coarse categories produces four well-separated clusters, each internally variable but categorically distinct. Coarse supervision does not merely yield a low-resolution version of the fine-grained representation; it gives rise to a qualitatively different organisation of the feature space. Sample images (*n* = 1,000 per panel) are drawn from ImageNet training images (Methods).

#### Observed results

- **(a) 1000-way model:** Smooth gradient in PC space — coarse labels are intermixed with no clear boundaries.
- **(b) 4-way model:** Four well-separated clusters, each internally variable but categorically distinct.
- **Cross-model RSA (supplementary):** Cross-model RSA (1000-way vs. coarse) increases from ρ ~ 0.22 (2-way) to ρ ~ 0.52 (64-way), but never approaches the inter-seed baseline (ρ ~ 0.76). Projection control confirms coarse features cannot be recovered by PCA of the fine-grained model (ρ ~ 0.03 at 2-class).

---

### Figure 3: Neural Alignment Across Species (TVSD + NSD)

**Directory:** `manuscript/figures/fig3/` · **Script:** `figure3.py`

#### Figure caption

**Fig. 3 | An extremely coarse feedback signal is sufficient for neural alignment across species and cortical hierarchy.**
**a**–**f**, Representational similarity analysis (RSA) comparing CNN activations with neural recordings from two independent datasets. Top row (**a**–**c**): macaque electrophysiology (TVSD, single-neuron spiking responses to object images, *n* = 2 monkeys). Bottom row (**d**–**f**): human functional magnetic resonance imaging (fMRI; NSD, *n* = 8 subjects viewing natural scenes). Columns correspond to early visual cortex (V1, early visual stream) and higher visual cortex (inferotemporal cortex (IT), ventral visual stream).
**a**,**d**, Experimental schematics showing stimulus types and recording modalities for TVSD (**a**) and NSD (**d**).
**b**,**c**,**e**,**f**, Each panel contains two vertically stacked sub-panels. *Top*: horizontal lollipop chart indicating the minimum number of coarse training classes (*k**) at which the model's 95% confidence interval first overlaps the 1000-class baseline (vertical dashed orange line), shown separately for AlexNet- and CLIP-derived coarse labels. *Bottom*: alignment (Spearman *ρ*) as a function of supervisory granularity (2–64 classes, log₂ scale), with a broken x-axis before the 1000-class position. Three sources of coarse labels are shown: labels derived from pretrained AlexNet representations (blue circles), from pretrained CLIP representations (dark blue squares) and from raw pixel values without any learned representation (tan triangles). The 1000-class supervised baseline is marked by an amber diamond; the untrained-network baseline by a grey dashed line.
**b**, Macaque V1: training on as few as two classes is sufficient to match 1000-class alignment, with near-constant performance across all granularity levels.
**c**, Macaque IT: alignment increases with granularity, but as few as eight classes suffice to reach the 1000-class baseline.
**e**, Human early visual stream: as few as two coarse classes produce alignment equivalent to 1000-class supervision, mirroring the pattern in macaque V1.
**f**, Human ventral visual stream: alignment increases with granularity, but as few as eight classes suffice to reach the 1000-class baseline, consistent with macaque IT. Pixel-based labels, which partition images by low-level statistics rather than semantic content, yield substantially lower alignment across all conditions, indicating that the relevant coarse structure must reflect high-level visual organisation.
Error bars denote 95% bootstrap confidence intervals aggregated across subjects (or monkeys) and three independently trained networks per condition.

#### Observed results

- **(b) TVSD V1:** AlexNet and CLIP at ρ ~ 0.16–0.18 across all levels, matching or exceeding 1000-way (ρ ~ 0.15). Flat curve — even 2-class matches.
- **(c) TVSD IT:** Monotonic ramp from ρ ~ 0.12 (2-class) to ρ ~ 0.17–0.19 (64-class), approaching 1000-way (ρ ~ 0.16).
- **(e) NSD Early Visual:** AlexNet and CLIP at ρ ~ 0.19–0.21 across all levels. 1000-way at ρ ~ 0.20.
- **(f) NSD Ventral Visual:** Ramp from ρ ~ 0.15 (2-class) to ρ ~ 0.25 (64-class). 1000-way at ρ ~ 0.21. CLIP converges fastest.

---

### Figure 4: Behavioral Alignment (THINGS)

**Directory:** `manuscript/figures/fig4/` · **Script:** `figure4.py`

#### Figure caption

**Fig. 4 | Coarse supervision produces the highest behavioural alignment of any model tested.**
**a**, Schematic of the THINGS behavioural benchmark. Human participants (*n* = 12,340) performed 4.70 million odd-one-out triplet judgements over 1,854 object concepts, yielding a 66-dimensional similarity embedding.
**b**, Alignment (Spearman *ρ*) with the THINGS embedding as a function of supervisory granularity (2–64 coarse classes, log₂ scale). Coarse labels are derived from three sources (AlexNet, CLIP and raw pixels); the 1,000-class baseline and untrained-network baseline are shown for reference. Models trained with CLIP- or AlexNet-derived coarse labels substantially exceed the 1,000-class baseline at every granularity level.
**c**, Comparison with pretrained models spanning supervised, self-supervised and vision–language training paradigms. The dashed line marks the best coarse-trained model (8 CLIP-derived classes). Despite being trained from scratch with only 8 categories, the coarse model matches or exceeds all pretrained systems.
**d**, Low-data regime. Coarse models trained on ~10,000 images (~1% of ImageNet) exceed the 1,000-class model trained on the full 1.2 million images.
**e**, Principal component projections (PC1 versus PC2) of concept-level representations for the THINGS behavioural embedding (ground truth), a CNN trained with 8 coarse classes, pretrained AlexNet (1,000 classes) and pretrained ViT-B/16 (1,000 classes). Points are coloured by six super-categories derived from the THINGS taxonomy; image insets highlight three example concepts across panels. The coarse-trained model recapitulates the categorical clustering of the behavioural ground truth, whereas 1,000-class models show more diffuse geometry.
Error bars in **b** and **d** denote 95% bootstrap confidence intervals (1,000 iterations) aggregated across three seeds.

#### Observed results

- **(b) Coarseness:** Headline result. CLIP labels strongest (ρ ~ 0.55–0.57), AlexNet (ρ ~ 0.44–0.48), all well above 1000-way (ρ ~ 0.39). Pixels start low (ρ ~ 0.10 at 2-class), never reach 1000-way. Untrained at ρ ~ 0.20.
- **(c) Model comparison:** Best coarse model (ρ ~ 0.57) exceeds 1000-way (ρ ~ 0.39). CLIP-L/14 and DINOv2 approach but do not exceed the coarse model.
- **(d) Data efficiency:** Coarse models on 10K images (1% of ImageNet) exceed 1000-way on the full 1.2M.
- **(e) PC scatter:** Coarse model mirrors behavioral ground-truth clustering; 1000-way models show more diffuse geometry.

#### Data notes

- Panel D uses legacy CSV at `experiments/coarse_grain_benefits/data_efficiency/legacy_results/data_efficiency_results.csv` (10K regime, not in the main results DB).

---

### Figure 5: Per-Concept Alignment Analysis

**Directory:** `manuscript/figures/fig5/` · **Script:** `figure5.py`

#### Figure caption

**Fig. 5 | The coarse-model advantage extends across semantic categories.**
**a**, Representational dissimilarity matrices (RDMs) for the THINGS behavioural embedding (ground truth), a CNN trained with 8 CLIP-derived classes and a CNN trained with 1,000 classes. Evaluation concepts (*n* = 380) are grouped into 10 super-categories (coloured sidebars) and sorted by hierarchical clustering within each group. The 8-class model ($\rho_s$ = 0.576) captures the block-diagonal structure of human similarity more faithfully than the 1,000-class model ($\rho_s$ = 0.392).
**b**, Mean per-concept alignment for each super-category, plotted for the 8-class model (x-axis) versus the 1,000-class model (y-axis). Points below the diagonal indicate categories where the coarse model achieves higher alignment. The majority of categories fall below the diagonal.
**c**, Distribution of per-concept advantage ($\Delta\rho_s$ = 8-class $-$ 1,000-class) across all evaluation concepts. Kernel density estimates are shown separately for all concepts, Living things and Body & apparel. 82% of individual concepts are better captured by the coarse model.

#### Observed results

- **(a) RDMs:** The 8-class model (ρ_s = 0.576) captures the block-diagonal structure much better than the 1000-class model (ρ_s = 0.392).
- **(b) Per-category scatter:** ~82% of concepts favor the coarse model. Living things consistently above the diagonal; Body & apparel below.
- **(c) Histogram:** Delta-ρ distribution is right-shifted. 82% favor coarse, 18% favor 1000-way. Positive tail is substantially longer.

#### Note

DPI is currently 200 in the script — should be updated to 300 per manuscript guidelines.

---

### Figure 6: Architecture Generalization (THINGS Behavioral)

**Directory:** `manuscript/figures/fig6/` · **Script:** `figure6.py`

#### Figure caption

**Fig. 6 | Coarse supervision produces more human-aligned representations regardless of architecture.**
**a**–**c**, Alignment (Spearman *ρ*) with the THINGS behavioural embedding as a function of supervisory granularity (2–64 CLIP-derived coarse classes, log₂ scale) for three architectures trained from scratch on ImageNet: ResNet-50 (**a**), ConvNeXt (**b**) and ViT-B/16 (**c**). The 1,000-class supervised baseline is shown as an amber diamond at the broken-axis position with a dashed reference line; the untrained-network baseline is shown as a grey dashed line. Across all three architectural families — spanning classic residual networks, modern convolutional designs and vision transformers — coarsely trained models substantially outperform their 1,000-class counterparts in behavioural alignment. The advantage is most pronounced for ConvNeXt and ViT-B/16, where coarse models exceed the fine-grained baseline by a wide margin, indicating that a coarse learning signal is sufficient to elicit human-aligned representations irrespective of model capacity, depth or inductive bias.

#### Observed results

- **(a) ResNet-50:** Coarse models at ρ ~ 0.55–0.58, above 1000-class (ρ ~ 0.50). Nearly flat curve.
- **(b) ConvNeXt:** Coarse models at ρ ~ 0.53–0.56, well above 1000-class (ρ ~ 0.35). Dramatic advantage.
- **(c) ViT-B/16:** Coarse models at ρ ~ 0.41–0.46, above 1000-class (ρ ~ 0.26). Proportionally the largest advantage (~60% improvement).
