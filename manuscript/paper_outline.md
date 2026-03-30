# Carving nature at its joints: a coarse feedback signal for learning human-aligned visual representations

## Paper Outline — Nature format, ~2,500 words main text

**Target:** Nature (main journal)
**Word limit:** ~2,500 words main text. Methods section is separate (Online Methods, unlimited). Supplementary figures S1–S18 provide supporting evidence.

---

## Structure Overview

```
Opening paragraphs                                    (~350 words, no heading)
├── P1: The question + field trajectory               (~150 words)
└── P2: "Here we show…"                              (~200 words)

§1  A modality-agnostic method for coarse             (~300 words → Fig 1)
    visual categories
├── P3: The coarse-graining method                    (~180 words)
└── P4: Different representations                     (~120 words)

§2  Coarse supervision suffices for neural            (~450 words → Fig 2)
    alignment across species
├── P5: TVSD — macaque electrophysiology              (~180 words)
├── P6: NSD — human fMRI                              (~170 words)
└── P7: Cross-species synthesis                       (~100 words)

§3  Coarse-trained models capture human               (~600 words → Figs 3 + 4)
    perceptual similarity better than
    fine-grained supervision
├── P8:  Coarseness curve (headline)                  (~150 words)
├── P9:  Pretrained model comparison                  (~130 words)
├── P10: Representational geometry                    (~120 words)
└── P11: Per-concept / RDM analysis                   (~200 words)

§4  The coarseness advantage generalizes              (~200 words → Fig 5)
    across architectures
└── P12: Three architectures, same result             (~200 words)

Closing paragraphs                                    (~350 words, no heading)
├── P13: Main conclusion + mechanism                  (~150 words)
├── P14: Broader implications                         (~120 words)
└── P15: Limitations + outlook                        (~80 words)

                                               TOTAL: ~2,250–2,500 words
```

---

## Opening paragraphs — no heading (~350 words)

### P1: The question + field trajectory (~150 words)

- Open with the broad question: what learning signals give rise to visual representations that resemble those in biological brains?
- Current direction in the field: progression toward richer supervision — from broad categories to 1000-class ImageNet to instance-level contrastive learning (SimCLR, DINO) and vision-language pretraining (CLIP)
- Implicit assumption: more detailed supervision → representations that better predict neural and behavioral responses
- This assumption has never been tested by *systematically varying* label granularity while holding everything else constant
- Gap: we don't know where on the granularity spectrum brain alignment actually peaks

### P2: "Here we show…" (~200 words)

- Introduce the approach in one sentence: we generate coarse category structures by applying PCA-based median splits in pretrained feature spaces, producing label sets ranging from 2 to 1000 classes
- Scale: hundreds of CNNs trained on ImageNet with varied granularity, identical architecture and procedure
- Three evaluation benchmarks, increasing in ecological validity:
  - Macaque single-neuron electrophysiology (TVSD) — V1, IT
  - Human fMRI (NSD) — early and ventral visual streams
  - Human behavioral similarity judgments (THINGS) — triplet odd-one-out
- State the three headline findings as crisp sentences:
  1. On neural data, coarse-trained networks (as few as 8–32 classes) match fully supervised 1000-class models in both species
  2. On behavioral similarity, coarse-trained networks *outperform* 1000-class supervision by ~40% and surpass state-of-the-art pretrained vision models
  3. These findings generalize across CNN, ConvNeXt, and Vision Transformer architectures
- Final sentence: concluding implication — the bottleneck for brain-like representations is not the richness of supervision but whether the label structure captures the right coarse partition of visual space

---

## §1 — A modality-agnostic method for coarse visual categories (~300 words → Fig 1)

### P3: The coarse-graining method (~180 words)

- Problem setup: how to systematically vary granularity while keeping the *type* of supervision constant (still category labels, just fewer categories)?
- Naive approaches fail: random grouping is arbitrary; WordNet hierarchy conflates semantic distance with granularity
- Our approach: project all 1000 ImageNet class centroids into a shared representation space (e.g., CLIP features), compute PCA, apply recursive median splits on the top principal components → 2^N classes (2, 4, 8, 16, 32, 64)
- **Fig 1a** reference: same 1000 points, same coordinates, different colorings reveal the progressive partitioning
- Key property: the method is modality-agnostic — any pretrained feature space can serve as the basis. We test three: CLIP, AlexNet, raw pixel averages (to show results don't depend on the source model)
- Training: standard CNN (AlexNet-style) trained on ImageNet with each label set, 3 seeds per condition. All other hyperparameters held constant. (Details in Methods)

### P4: Coarse training produces qualitatively different representations (~120 words)

- **Fig 1b** reference: PCA of learned FC1 activations
- 1000-way model: smooth gradient in PC space — representations organized to separate 1000 fine classes, coarse structure dissolved
- 4-way model: four well-separated clusters with clear categorical boundaries
- This is NOT recoverable by dimensionality reduction of the fine-grained model — cross-model RSA between coarse-trained and projected-1000-way representations is extremely low (ρ ~ 0.03 at 2-class; Supplementary)
- Key point: coarse training fundamentally *reorganizes* representational geometry, not just reduces it

---

## §2 — Coarse supervision suffices for neural alignment across species (~450 words → Fig 2)

### P5: Macaque electrophysiology — TVSD (~180 words)

- Evaluation method in one sentence: representational similarity analysis (RSA) — compare model and neural representational dissimilarity matrices via Spearman correlation
- TVSD dataset: single-neuron recordings from two macaques viewing ~22K natural images, spanning V1 (early) and IT (higher visual cortex)
- **Fig 2b** — V1: coarse models match or exceed 1000-way at ALL granularity levels (ρ ~ 0.16–0.18 across 2–64 classes vs. 1000-way ρ ~ 0.15). Curve is flat
- Interpretation: the basic visual features encoded in V1 (edges, spatial frequencies, textures) emerge from any categorization objective, however coarse
- **Fig 2c** — IT: monotonic increase from 2-class (ρ ~ 0.12) to 64-class (ρ ~ 0.17–0.19), approaching 1000-way (ρ ~ 0.16). More category structure is needed for IT-level object representations, but saturation occurs well below 1000 classes
- Untrained baseline is clearly separated in both regions, confirming that category-supervised training matters — but the *amount* of supervision barely matters

### P6: Human fMRI — NSD (~170 words)

- NSD dataset: 8 human subjects viewing ~10K natural scenes, voxel-level fMRI responses in early visual stream (V1/V2/V3) and ventral visual stream (VO, PHC, higher areas)
- **Fig 2e** — Early visual stream: replicates V1 pattern. All coarse models at ρ ~ 0.19–0.21, matching 1000-way (ρ ~ 0.20)
- **Fig 2f** — Ventral visual stream: gradual ramp from ρ ~ 0.15 (2-class) to ρ ~ 0.25 (64-class). 1000-way at ρ ~ 0.21 — notably, 32–64 class models *exceed* the fully supervised baseline
- Results hold across all three PCA source models; CLIP labels converge fastest

### P7: Cross-species synthesis (~100 words)

- Remarkable consistency: same saturation pattern across species (macaque, human), measurement modalities (single-neuron, fMRI), and visual hierarchy (early → higher cortex)
- The functional hierarchy is preserved: early regions are insensitive to granularity; higher regions show graded sensitivity but saturate at 32–64 classes
- Bottom line: fine-grained category supervision provides negligible additional benefit for neural alignment. The representational features that predict brain activity emerge from coarse categorical structure alone
- Additional controls (encoding score analysis, per-layer profiles, stimulus robustness) in Supplementary confirm these findings

---

## §3 — Coarse-trained models capture human perceptual similarity better than fine-grained supervision (~600 words → Figs 3 + 4)

### P8: The coarseness curve — headline result (~150 words)

- THINGS benchmark: human behavioral similarity from 4.7M triplet odd-one-out judgments across 1,854 object concepts. The gold standard for human perceptual similarity structure
- **Fig 3b**: ALL coarse-trained models sit dramatically *above* the 1000-way baseline — the opposite of what the field would predict
- CLIP-derived labels: ρ ~ 0.55–0.57 vs. 1000-way ρ ~ 0.39. Even 2-class models exceed 1000-way
- AlexNet labels: ρ ~ 0.44–0.48, still well above 1000-way
- This is not a subtle effect — it's a ~40% improvement over full ImageNet supervision
- The untrained baseline (ρ ~ 0.20) confirms this is learned, not architectural
- Pixel-derived labels are weaker (ρ ~ 0.10–0.23), indicating that the *quality* of the coarse partition matters, but even pixel-based labels approach 1000-way performance at higher granularities

### P9: Pretrained model comparison (~130 words)

- **Fig 3c**: How does the best coarse model (8-class, CLIP labels, trained from scratch) compare against pretrained models with orders of magnitude more data and compute?
- It matches or exceeds all supervised pretrained CNNs (AlexNet, VGG-16, ResNet-50, ConvNeXt)
- It matches or exceeds self-supervised models (DINOv1, DINOv2)
- Only CLIP-L/14 (trained on 400M image-text pairs) approaches the coarse model's score — and does not clearly surpass it
- Key framing: a simple CNN trained from scratch on 8 categories rivals the best vision models in capturing human similarity. This challenges the assumption that scale and sophisticated objectives are necessary

### P10: Representational geometry explains the advantage (~120 words)

- **Fig 3d**: PCA projections of THINGS concept representations, colored by 8 semantic super-categories (Animal, Vehicle, Food, etc.)
- Behavioral ground truth: clear super-category clustering — humans judge similarity primarily by broad category membership
- CNN 8-class (CLIP): reproduces this broad categorical separation remarkably well
- AlexNet-1K and ViT-B/16-1K: more diffuse, fragmented geometry — fine-grained training pushes apart within-category exemplars, dissolving the between-category structure that dominates human judgments
- This visual argument is quantified in the per-concept analysis below

### P11: Per-concept and RDM analysis (~200 words)

- **Fig 4a** — Category-sorted RDMs: when concepts are grouped by 10 semantic super-categories, the behavioral RDM shows prominent block-diagonal structure. The 8-class model captures this block structure (ρ = 0.576); the 1000-way model shows weaker block boundaries (ρ = 0.392). Fine-grained training over-differentiates within categories at the expense of between-category contrast
- **Fig 4b** — Per-concept scatter: 82% of individual concepts show higher alignment for the coarse model than the 1000-way model. The advantage spans most semantic categories (plants, animals, clothing, tools). The minority where 1000-way wins (body parts, drinks) are interpretable — these contain fine-grained within-category distinctions that humans *do* weight in similarity
- **Fig 4c** — Advantage distribution: the per-concept advantage (Δρ) is broadly right-shifted, confirming this is a pervasive effect across the concept space, not driven by outlier categories
- Mechanistic interpretation: coarse labels force networks to learn features that distinguish broad natural categories — the very structure that dominates human perceptual similarity. 1000-class labels incentivize sub-category distinctions (breeds, species, model variants) that humans de-emphasize

---

## §4 — The coarseness advantage generalizes across architectures (~200 words → Fig 5)

### P12: Three architectures, same pattern (~200 words)

- A natural concern: is the coarseness advantage specific to the AlexNet-style CNN used above?
- **Fig 5**: replicate the THINGS experiment with three diverse architectures — ResNet-50 (deep residual CNN), ConvNeXt (modern CNN), and ViT-B/16 (vision transformer). All trained from scratch with CLIP-derived coarse labels
- ResNet-50: coarse ρ ~ 0.55–0.58, 1000-way ρ ~ 0.50 — clear coarse advantage
- ConvNeXt: coarse ρ ~ 0.53–0.56, 1000-way ρ ~ 0.35 — dramatic advantage
- ViT-B/16: coarse ρ ~ 0.41–0.46, 1000-way ρ ~ 0.26 — proportionally the largest advantage (~60%)
- The finding is architecture-independent: CNNs, modern CNNs, and transformers all show the same pattern
- Results also hold across PCA source models (Supplementary), ruling out an interaction between the label-generation architecture and the trained architecture
- This robustness strengthens the central claim: the coarseness advantage reflects a fundamental property of how category structure shapes learned representations, not an architectural quirk

---

## Closing paragraphs — no heading (~350 words)

### P13: Main conclusion + mechanistic insight (~150 words)

- Restate the core finding without repeating the opening: across species, measurement modalities, and architectures, coarse category supervision produces representations that are as aligned with biological vision as — or more aligned than — fine-grained 1000-class supervision
- The mechanism: coarse labels force networks to learn features that distinguish broad semantic categories (animate vs. inanimate, natural vs. man-made). These broad distinctions correspond to the dominant axes of human perceptual similarity. Fine-grained labels dilute this structure by incentivizing sub-category discrimination (e.g., 120 dog breeds) that humans de-emphasize in similarity judgments
- Connect to the title: the network discovers how to "carve nature at its joints" — the coarse categorical boundaries that organize human visual perception

### P14: Broader implications (~120 words)

- For neuroscience / development: the learning signals that shape biological visual representations may be far coarser than previously assumed. Infants' early visual categories are broad (animal, vehicle, face) before differentiating — consistent with our finding that coarse structure suffices
- For AI alignment: current efforts to improve brain-model alignment focus on scaling data and supervision. Our results suggest an orthogonal strategy: identify the *right coarse structure* of the training signal. This could be more data-efficient (reference Supplementary data-efficiency analysis if available)
- For representation learning more broadly: the modality-agnostic coarse-graining method may generalize beyond vision to auditory, linguistic, or multimodal domains

### P15: Limitations + outlook (~80 words)

- Current study uses static images and category-level supervision; extending to video, temporal dynamics, and other supervision types (language, reward signals) is an important next step
- The optimal granularity may depend on the downstream task — our results show alignment with human *perceptual similarity*, not necessarily with other cognitive functions (naming, reasoning)
- Whether the PCA-based coarse structure corresponds to cognitively real categories (basic-level, superordinate) warrants direct investigation

---

## Key Structural Decisions

1. **No traditional IMRaD.** Nature format uses opening paragraphs → results with declarative subheadings → closing paragraphs. Methods go entirely in Online Methods (separate, unlimited word count).

2. **Figs 3 + 4 merged into one section.** They answer the same question: "does coarse win on behavioral similarity, and if so why?" Splitting them wastes transition words and fragments the story.

3. **Fig 5 is standalone.** Short section (~200 words) with a declarative heading. Architecture generalization is the final piece of evidence — it closes the empirical argument before the conceptual closing.

4. **Declarative headings.** Each heading states the result, not the method. A reader who skims only the headings gets the full story.

5. **"Here we show" paragraph is self-contained.** A reviewer who reads only the abstract and P2 should understand the full claim and evidence structure.

## What Goes Where

| Content | Location |
|---------|----------|
| PCA label generation algorithm | Methods |
| CNN architecture details | Methods |
| Training hyperparameters, optimizer, epochs | Methods |
| RSA pipeline (SRP, layer selection, bootstrap) | Methods |
| Encoding score pipeline | Methods |
| Dataset details (NSD, TVSD, THINGS) | Methods |
| Statistical tests, confidence intervals | Methods |
| V4 (TVSD) results | Supplementary |
| NSD-Synthetic results | Supplementary |
| DINO / ViT PCA label results | Supplementary |
| Per-layer profiles (all granularities) | Supplementary |
| Encoding score analysis | Supplementary |
| Cross-model RSA controls | Supplementary |
| Stimulus robustness analysis | Supplementary |
| Seed variability analysis | Supplementary |
| Fine-grained ROI breakdown (6 NSD ROIs) | Supplementary |
| Data efficiency analysis | Supplementary |
| WordNet-based label comparison | Supplementary |
