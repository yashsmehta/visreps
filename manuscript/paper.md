# An extremely coarse feedback signal is sufficient for learning human-aligned visual representations
**Abstract.** Artificial neural networks trained on visual tasks develop internal representations resembling those of the primate visual system, a discovery that has guided a decade of computational neuroscience. A prevailing assumption in the field is that learning fine-grained distinctions is necessary (or at least beneficial) for producing brain-like representations. Accordingly, the field of neuroAI has moved toward increasingly granular training signals, from thousand-way object classification to self-supervised objectives that treat every image as its own class. Yet this assumption remains untested. Here we ask how coarse a learning signal can be and still give rise to human-aligned visual representations. We develop a data-driven, modality-agnostic method to partition visual inputs into categories at varying levels of granularity (2, 4, 8, 16, … 64) and train hundreds of neural networks across diverse architectures on these coarse classification tasks. Networks trained to distinguish as few as 8 broad categories learn representations that match or exceed the neural alignment of 1,000-class supervised models, measured against macaque single-neuron recordings and human fMRI responses. These coarsely trained networks align more closely with human perceptual similarity judgments than every other model we tested — including fine-grained supervised networks, self-supervised methods, and billion-parameter systems like CLIP and DINOv3 — regardless of architecture, training paradigm, or dataset scale. These results demonstrate that human-like visual representations emerge from remarkably coarse feedback, challenging the field's drive toward ever more granular objectives and reframing what learning signals the visual system may require.

---

## Paper structure (Nature format)

Main text ~4000 words | Methods ~3000 words | Extended Data up to 10 figures | Supplementary Information (no figures)

---

## Core method: coarse-graining the input space

- Extract features of all ImageNet training images from a pretrained source network (CLIP, AlexNet, ViT, DINOv3, or raw pixels).
- Compute PCA of this representation space.
- PC1 median split → 2 classes. PC2 splits each half → 4 classes. Recurse: 2^n classes after n splits.
- Each image gets a discrete label from K coarse categories; supervisory signal = at most log₂(K) bits/image.
- Labels group up to ~500K diverse images per class.
- Train new networks **from scratch** at each granularity (2, 4, 8, 16, 32, 64, 1000), holding architecture, dataset, augmentation, LR, normalization, and all hyperparameters constant. Sole experimental variable = supervisory granularity.
- Source model contributes **only category assignments**: no feature transfer, no distillation, no shared gradients.

## Evaluation: representational similarity analysis (RSA)

- Construct RDMs (pairwise distance matrices) from model activations and from biological measurements on shared stimulus sets.
- Alignment = Spearman ρ between model RDM and target RDM.
- Three evaluation targets: (1) macaque single-neuron spiking (TVSD, V1/V4/IT, 2 monkeys, >25K images), (2) human fMRI (NSD, 8 subjects, ~10K images, early + ventral visual stream), (3) human behavioral similarity (THINGS, 12,340 participants, 4.70M odd-one-out triplets, 1,854 concepts, 66-dim embedding).

## Architectures tested

AlexNet-like CNN (5 conv + 3 FC), ResNet-50, ConvNeXt, ViT-B/16. All trained from scratch on ImageNet.

## Benchmark models

Pretrained supervised (AlexNet, ResNet-50, VGG-16, ViT-B/16), self-supervised (DINOv1, DINOv2, DINOv3), vision-language (CLIP-B/32, CLIP-L/14).

---

## Key findings

**1. Neural alignment (Fig. 3).** 8 coarse classes match or exceed 1000-class neural alignment across species and cortical hierarchy. Macaque V1: even 2 classes suffice. Macaque IT / human ventral stream: 8 classes suffice. Pattern holds for both CLIP-derived and AlexNet-derived labels.

**2. Behavioral alignment — headline result (Fig. 4).** Coarse models **substantially exceed** 1000-class models on THINGS behavioral similarity (not just match). Even 2 classes improve over 1000-class; peak at ~8 classes. 8-class CLIP-derived model outperforms **every** pretrained model tested (CLIP, DINOv3, DINOv2, all supervised baselines) regardless of architecture, paradigm, or dataset scale.

**3. Qualitatively different representations (Fig. 2).** Coarse training does not compress fine-grained geometry; it produces a fundamentally different representational structure with stronger categorical clustering matching human perceptual organization.

**4. Per-concept decomposition (Fig. 5).** 82% of individual THINGS concepts better captured by 8-class model than 1000-class. Advantage spans all 10 super-categories, not confined to easily separable domains. 8-class RDM ρ = 0.576 vs. 1000-class ρ = 0.392.

**5. Low-data regime (Fig. 4d).** Coarse models trained on ~10K images (1% of ImageNet) exceed 1000-class model trained on full 1.2M images.

**6. Architecture generality (Fig. 6).** Pattern holds for ResNet-50, ConvNeXt, ViT-B/16. Advantage most pronounced for ConvNeXt and ViT.

**7. Pixel-based labels fail.** Categories from raw pixel PCA (low-level features like luminance/contrast) yield no substantial alignment. The coarse structure must reflect high-level visual organization.

**8. Source model independence.** Similar trends for CLIP, AlexNet, ViT, DINOv3 as label sources (main figures show CLIP + AlexNet; supplementary shows others).

---

## Key negative / null result

Coarse training did **not** improve robustness to image corruption vs. fine-grained models. Perceptual alignment and standard CV benchmarks measure dissociable properties.

## Central theoretical claim

What matters is whether the feedback signal partitions the input space along perceptually relevant boundaries, not how many bits it carries. This reframes the learning signal question for biological vision: coarse categorical feedback may be more appropriate than fine-grained supervision.

## Scope / limitations noted

- Method requires a partitionable feature space; extensible in principle to auditory/tactile/multimodal (left to future work).
- Mechanism producing coarse-grained bias in biological cortex remains an open question.
- No corruption robustness benefit.