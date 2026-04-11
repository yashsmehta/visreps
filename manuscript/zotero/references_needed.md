# Candidate references for the manuscript

Compiled from `paper.md`, `methods.md`, `supplementary_information.md`, and Extended Data captions. **Unverified — must be checked online (DOI, venue, year, author order) before any Zotero import.** Grouped by the claim or section that would cite them.

Each reference is tagged with a verdict:
- **ESSENTIAL** — directly cited or load-bearing for a specific claim. Must be in the final bibliography.
- **KEEP** — relevant and worth importing to Zotero now; may be trimmed during final editing.
- **REMOVE** — not relevant enough to warrant importing. Rationale given.

---

## 1. CNNs as models of the primate visual system (opening premise)

- **ESSENTIAL** · Yamins, Hong, Cadieu, Solomon, Seibert, DiCarlo (2014). *Performance-optimized hierarchical models predict neural responses in higher visual cortex.* **PNAS**. — The foundational result: DNNs predict IT responses.
- **ESSENTIAL** · Khaligh-Razavi & Kriegeskorte (2014). *Deep supervised, but not unsupervised, models may explain IT cortical representation.* **PLoS Comp Bio**. — First systematic DNN-vs-brain comparison. [Zotero: `T22MPRXV`]
- **KEEP** · Güçlü & van Gerven (2015). *Deep neural networks reveal a gradient in the complexity of neural representations across the ventral stream.* **J Neurosci**. — Gradient across the ventral stream, directly relevant to the early-vs-higher cortex split. [Zotero: `2DH23EKH`]
- **KEEP** · Cadieu et al. (2014). *Deep neural networks rival the representation of primate IT cortex for core visual object recognition.* **PLoS Comp Bio**. — Converging evidence with Yamins 2014; useful for the opening multi-cite but can be trimmed if space-constrained.
- ~~REMOVE~~ · ~~Kriegeskorte (2015). *Deep neural networks: a new framework for modeling biological vision and brain information processing.* **Annual Review of Vision Science**.~~ — Review paper superseded by Yamins & DiCarlo 2016.
- **ESSENTIAL** · Yamins & DiCarlo (2016). *Using goal-driven deep learning models to understand sensory cortex.* **Nature Neuroscience**. — The authoritative review of goal-driven DNN models for sensory cortex.
- **ESSENTIAL** · Schrimpf, Kubilius, Hong, Majaj, Rajalingham, Issa, Kar, Bashivan, Prescott-Roy, Schmidt, Minces, DiCarlo (2020). *Integrative benchmarking to advance neurally mechanistic models of human intelligence.* **Neuron** 108(3):413–423. doi:10.1016/j.neuron.2020.07.040. — Brain-Score benchmarking framework. **NOTE:** `methods.md` currently cites the 2018 bioRxiv preprint; should be updated to this 2020 Neuron publication.
- **ESSENTIAL** · Conwell, Prince, Kay, Alvarez, Konkle (2024). *A large-scale examination of inductive biases shaping high-level visual representation in brains and machines.* **Nature Communications**. — Already cited in methods. Most recent large-scale brain-model alignment study.
- **KEEP** · Doerig et al. (2023). *The neuroconnectionist research programme.* **Nature Reviews Neuroscience**. — Broad review; useful for the opening paragraph but can be trimmed.
- **KEEP** · Storrs, Kietzmann, Walther, Mehrer & Kriegeskorte (2021). *Diverse deep neural networks all predict human inferior temporal cortex well, after training and fitting.* **J Cogn Neurosci** 33(10):2044–2064. — Shows many architectures predict IT; directly sets up the question of *why* (is it the fine-grained objective, or something else?). **[NEW]**
- **KEEP** · Elmoznino & Bonner (2024). *High-performing neural network models of visual cortex benefit from high latent dimensionality.* **PLoS Comp Bio** 20:e1011792. — Higher effective dimensionality predicts cortical alignment. Relevant to ED Fig. 6 reconstruction control and the dimensionality discussion. **[NEW — 2024]**
- **KEEP** · Kazemian, Elmoznino & Bonner (2025). *Convolutional architectures are cortex-aligned de novo.* **Nature Machine Intelligence**. — Untrained CNNs predict visual cortex nearly as well as trained ones; challenges the view that training is the primary driver of brain alignment. Directly relevant: if architecture alone gets you partway, your paper shows how *coarse* training gets you the rest. **[NEW — 2025]**

## 2. Self-supervised learning as "every image is its own class" (motivation)

- **KEEP** · Chen, Kornblith, Norouzi, Hinton (2020). *A simple framework for contrastive learning of visual representations (SimCLR).* **ICML**. — Canonical contrastive SSL reference for the abstract's "every image is its own class" framing.
- ~~REMOVE~~ · ~~He, Fan, Wu, Xie, Girshick (2020). *Momentum contrast for unsupervised visual representation learning (MoCo).* **CVPR**.~~ — SimCLR is sufficient to establish the contrastive SSL paradigm. Not used or evaluated.
- ~~REMOVE~~ · ~~Grill et al. (2020). *Bootstrap Your Own Latent (BYOL).* **NeurIPS**.~~ — Not used or discussed.
- ~~REMOVE~~ · ~~Chen & He (2021). *Exploring simple siamese representation learning (SimSiam).* **CVPR**.~~ — Not used or discussed.
- **KEEP** · Caron et al. (2021). *Emerging properties in self-supervised vision transformers (DINO / DINOv1).* **ICCV**. — DINOv1 is a benchmark model evaluated in the paper.
- **ESSENTIAL** · Oquab et al. (2023). *DINOv2: Learning robust visual features without supervision.* **TMLR / arXiv**. — Benchmark model AND used as PCA label source. Already cited in methods.
- **ESSENTIAL** · Siméoni, Vo, Seitzer, Baldassarre, Oquab, et al. (2025). *DINOv3: Self-Supervised Learning for Vision at Unprecedented Scale.* **arXiv:2508.10104** (Meta AI, 13 Aug 2025). — Used as PCA label source AND benchmark. No peer-reviewed venue yet; cite arXiv ID. ⚠️ **CITATION FIX NEEDED:** `methods.md` currently cites "Oquab et al., 2024" for DINOv3 — that's the DINOv2 paper. DINOv3 is Siméoni et al. 2025. Both citations are needed.
- **ESSENTIAL** · Zhuang et al. (2021). *Unsupervised neural network models of the ventral visual stream.* **PNAS**. — Shows unsupervised models match supervised ones for brain alignment — the premise your paper refines.
- **ESSENTIAL** · Konkle & Alvarez (2022). *A self-supervised domain-general learning framework for human ventral stream representation.* **Nature Communications**. — Most directly relevant prior work: SSL → brain alignment.
- **KEEP** · Margalit, Lee, Finzi, DiCarlo, Grill-Spector & Yamins (2024). *A unifying framework for functional organization in early and higher ventral visual cortex.* **Neuron**. doi:10.1016/j.neuron.2024.04.018. — TDANN: self-supervised objective yields more brain-like functional organization than supervised classification. Relevant to the argument that the *type* of training objective shapes alignment. **[NEW — 2024]**

## 3. Vision-language / large-scale pretraining (benchmark models)

- **ESSENTIAL** · Radford et al. (2021). *Learning transferable visual models from natural language supervision (CLIP).* **ICML**. — Already cited. Label source + benchmark.
- ~~REMOVE~~ · ~~Jia et al. (2021). *Scaling up visual and vision-language representation learning with noisy text supervision (ALIGN).* **ICML**.~~ — Not used anywhere. CLIP is the only VL model in the paper.

## 4. Architectures trained from scratch

- **ESSENTIAL** · Krizhevsky, Sutskever, Hinton (2012). *ImageNet classification with deep convolutional neural networks (AlexNet).* **NeurIPS**. — Already cited. Primary architecture.
- **KEEP** · Simonyan & Zisserman (2015). *Very deep convolutional networks for large-scale image recognition (VGG).* **ICLR**. — VGG-16 is a benchmark model.
- **ESSENTIAL** · He, Zhang, Ren, Sun (2016). *Deep residual learning for image recognition (ResNet).* **CVPR**. — Already cited. Architecture generalization (Fig. 6).
- **ESSENTIAL** · Liu, Mao, Wu, Feichtenhofer, Darrell, Xie (2022). *A ConvNet for the 2020s (ConvNeXt).* **CVPR**. — Already cited. Architecture generalization (Fig. 6).
- **ESSENTIAL** · Dosovitskiy et al. (2021). *An image is worth 16x16 words: transformers for image recognition at scale (ViT).* **ICLR**. — Already cited. Architecture + label source.
- ~~REMOVE~~ · ~~Vaswani et al. (2017). *Attention is all you need.* **NeurIPS**.~~ — Citing ViT is sufficient. Not needed for a paper about visual representations.

## 5. Training data

- **ESSENTIAL** · Deng, Dong, Socher, Li, Li, Fei-Fei (2009). *ImageNet: A large-scale hierarchical image database.* **CVPR**. — Already cited.
- ~~REMOVE~~ · ~~Russakovsky et al. (2015). *ImageNet Large Scale Visual Recognition Challenge (ILSVRC).* **IJCV**.~~ — Redundant with Deng 2009. The paper uses ImageNet-1K, not the ILSVRC challenge.

## 6. Representational Similarity Analysis (evaluation method)

- **ESSENTIAL** · Kriegeskorte, Mur, Bandettini (2008). *Representational similarity analysis — connecting the branches of systems neuroscience.* **Frontiers in Systems Neuroscience**. — Already cited. The RSA paper.
- ~~REMOVE~~ · ~~Kriegeskorte & Kievit (2013). *Representational geometry: integrating cognition, computation, and the brain.* **Trends in Cognitive Sciences**.~~ — Framework paper; the 2008 paper covers RSA.
- ~~REMOVE~~ · ~~Nili et al. (2014). *A toolbox for representational similarity analysis.* **PLoS Comp Bio**.~~ — Methods explicitly state RSA is "implemented from scratch without reliance on existing RSA toolboxes." Citing the toolbox paper would be contradictory.

## 7. Neural and behavioral datasets (evaluation targets)

**Macaque electrophysiology (TVSD):**
- **ESSENTIAL** · Papale et al. (2025). *An extensive dataset of spiking activity to reveal the syntax of the ventral stream.* **Neuron** 113(4):539–553.e5. doi:10.1016/j.neuron.2024.12.003. — TVSD: 2 macaques, V1/V4/IT, THINGS stimuli.
- **ESSENTIAL** · Hebart et al. (2019). *THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images.* **PLoS ONE**. — Already cited. TVSD uses THINGS images.

**Human fMRI (NSD):**
- **ESSENTIAL** · Allen et al. (2022). *A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence (Natural Scenes Dataset).* **Nature Neuroscience**. — Already cited.
- **ESSENTIAL** · Lin et al. (2014). *Microsoft COCO: Common objects in context.* **ECCV**. — Already cited in methods. NSD stimuli come from COCO. **[Previously missing from this list]**
- **ESSENTIAL** · Prince, Charest, Bhatt, Hutchinson, Scholl, Kay (2022). *Improving the accuracy of single-trial fMRI response estimates using GLMsingle.* **eLife**. — Already cited in methods. NSD preprocessing pipeline. **[Previously missing from this list]**

**Human behavior (THINGS):**
- **ESSENTIAL** · Hebart et al. (2020). *Revealing the multidimensional mental representations of natural objects underlying human similarity judgments.* **Nature Human Behaviour** 4(11):1173–1185. doi:10.1038/s41562-020-00951-3. — THINGS odd-one-out + SPoSE embedding. Already cited.
- **ESSENTIAL** · Hebart et al. (2023). *THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior.* **eLife**. — Already cited.
- **ESSENTIAL** · Zheng, Pereira, Baker, Hebart (2019). *Revealing interpretable object representations from human behavior.* **ICLR**. (SPoSE.) — Already cited. The method for deriving behavioral embeddings.
- ~~REMOVE~~ · ~~Muttenthaler et al. (2022). *Human alignment of neural network representations.* **ICLR 2023**.~~ — Not used in the paper. Relevant reading but not load-bearing.
- ~~REMOVE~~ · ~~Muttenthaler, Zheng, McClure, Vandermeulen, Hebart, Pereira (2022). *VICE: Variational Interpretable Concept Embeddings.* **NeurIPS 2022**.~~ — Not used. Paper uses SPoSE, not VICE.

## 8. Category structure of human vision (interpretive framing for Figs. 2, 4, 5)

- **ESSENTIAL** · Rosch et al. (1976). *Basic objects in natural categories.* **Cognitive Psychology**. — Foundational reference for category levels in cognition. The paper argues coarse categories suffice — Rosch is the classic framing.
- **KEEP** · Grill-Spector & Weiner (2014). *The functional architecture of the ventral temporal cortex and its role in categorization.* **Nature Reviews Neuroscience**. — Relevant for interpreting Fig. 3's ventral stream results.
- **KEEP** · Kriegeskorte et al. (2008). *Matching categorical object representations in inferior temporal cortex of man and monkey.* **Neuron**. — Categorical representations in IT across species — directly relevant to the cross-species comparison in Fig. 3.
- ~~REMOVE~~ · ~~Huth, Nishimoto, Vu, Gallant (2012). *A continuous semantic space describes the representation of thousands of object and action categories across the human brain.* **Neuron**.~~ — Argues for a *continuous* space, the opposite of the categorical framing. Including it muddies the argument.
- **KEEP** · Konkle & Caramazza (2013). *Tripartite organization of the ventral stream by animacy and object size.* **J Neurosci**. — Organization by animacy/size. Fig. 5's super-category results connect to this.
- ~~REMOVE~~ · ~~Mahon & Caramazza (2011). *What drives the organization of object knowledge in the brain?* **Trends in Cognitive Sciences**.~~ — Theoretical review; Konkle & Caramazza 2013 is more specific and sufficient.
- **KEEP** · Cichy, Khosla, Pantazis, Torralba & Oliva (2016). *Comparison of deep neural networks to spatio-temporal cortical dynamics of human visual object recognition reveals hierarchical correspondence.* **Scientific Reports** 6:27755. doi:10.1038/srep27755. — Temporal dynamics of categorical representation in visual cortex; training on categorization is necessary for hierarchical correspondence. **[NEW]**
- **KEEP** · Mehrer, Kietzmann & Kriegeskorte (2021). *An ecologically motivated image dataset for deep learning yields better models of human vision.* **PNAS** 118(8):e2011417118. — ecoset: training data composition affects brain alignment — adjacent to the claim that the *objective* matters. **[NEW]**
- **KEEP** · Badwal, Bergmann, Roth, Doeller & Hebart (2025). *The scope and limits of fine-grained image and category information in the ventral visual pathway.* **J Neurosci** 45(3):e0936242024. — Shows fine-grained category effects in LOC are subtle while image-specific effects dominate. Directly relevant to interpreting your finding that coarse categories suffice. **[NEW — 2025]**

## 9. Coarse vs. fine-grained supervision / label granularity (directly adjacent work)

- **ESSENTIAL** · Geirhos et al. (2021). *Partial success in closing the gap between human and machine vision.* **NeurIPS**. — Documents the gap between human and machine vision. Coarse models partially close it.
- **KEEP** · Bracci, Ritchie, Kalfas, Op de Beeck (2019). *The ventral visual pathway represents animal appearance over animacy, unlike human behavior and deep neural networks.* **J Neurosci**. — How DNNs differ from biological vision in category representation.
- ~~REMOVE~~ · ~~Xie, Girshick, Dollár, Tu, He (2017). *Aggregated residual transformations for deep neural networks (ResNeXt).* — Architecture paper; nothing to do with label granularity.~~
- ~~REMOVE~~ · ~~Sun, Shrivastava, Singh, Gupta (2017). *Revisiting unreasonable effectiveness of data in deep learning era (JFT-300M).* **ICCV**.~~ — About data scale, not label granularity.
- ~~REMOVE~~ · ~~Mahajan et al. (2018). *Exploring the limits of weakly supervised pretraining.* **ECCV**.~~ — Weakly supervised with hashtags; interesting parallel but not discussed in the paper.
- ~~REMOVE~~ · ~~Kornblith, Shlens, Le (2019). *Do better ImageNet models transfer better?* **CVPR**.~~ — Transfer learning ≠ brain alignment.
- **KEEP** · Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* **ICLR**. — Shape vs texture bias. Relevant to interpreting *why* coarse representations differ; connects to the pixel-label null result.
- **KEEP** · Fel, Rodriguez, Linsley & Serre (2022). *Harmonizing the object recognition strategies of deep neural networks with humans.* **NeurIPS 2022**. — Shows a systematic trade-off between DNN accuracy and alignment with human visual strategies; proposes a "neural harmonizer" to fix this. Relevant to the discussion. **[NEW]**
- **KEEP** · Muttenthaler, Linhardt, Dippel, Vandermeulen, Hermann, Lampinen & Kornblith (2023). *Improving neural network representations using human similarity judgments.* **NeurIPS 2023**. — Uses THINGS-like behavioral data to align DNN global representational structure with humans. Adjacent work: they post-hoc align representations; you show coarse training achieves this inherently. **[NEW]**
- **ESSENTIAL** · Elmoznino, Vieira, Deng, Bonner et al. (2025). *Aligning machine and human visual representations across abstraction levels.* **Nature** 647:349–355. doi:10.1038/s41586-025-09631-6. — **HIGHLY RELEVANT.** Shows that model representations fail specifically at coarse-grained/global abstraction levels, and proposes fine-tuning on a hierarchical "Levels" dataset to fix this. Your paper provides a complementary finding: training *from scratch* with coarse labels inherently produces better-aligned representations. The two papers attack the same problem from opposite directions. **[NEW — 2025, user-requested]**
- **KEEP** · Mahner, Muttenthaler, Güçlü & Hebart (2025). *Dimensions underlying the representational alignment of deep neural networks with humans.* **Nature Machine Intelligence** 7:848–859. doi:10.1038/s42256-025-01041-7. — Identifies latent dimensions of human-DNN alignment; shows DNNs have visual > semantic dominance unlike humans. Relevant to interpreting *why* coarse training shifts the balance toward more human-like structure. **[NEW — 2025]**

## 10. Robustness / corruption benchmarks (null result section)

- **KEEP** · Hendrycks & Dietterich (2019). *Benchmarking neural network robustness to common corruptions and perturbations (ImageNet-C).* **ICLR**. — Needed if the null result cites ImageNet-C.
- ~~REMOVE~~ · ~~Geirhos, Temme, Rauber, Schütt, Bethge, Wichmann (2018). *Generalisation in humans and deep neural networks.* **NeurIPS**.~~ — One reference is enough for a null result.

## 11. Encoding models (Extended Data Fig. 3)

- **KEEP** · Naselaris, Kay, Nishimoto, Gallant (2011). *Encoding and decoding in fMRI.* **NeuroImage**. — Canonical encoding model reference. Used in ED Fig. 3.
- ~~REMOVE~~ · ~~Kay, Naselaris, Prenger, Gallant (2008). *Identifying natural images from human brain activity.* **Nature**.~~ — Foundational but Naselaris 2011 is more directly relevant.

## 12. WordNet labels (Supplementary Note 2)

- **KEEP** · Miller (1995). *WordNet: A lexical database for English.* **Communications of the ACM**. — Referenced in Supplementary Note 2 for WordNet-derived labels. **[Previously missing from this list]**
- **KEEP** · Wu & Palmer (1994). *Verb semantics and lexical selection.* **ACL**. — Wu–Palmer semantic similarity used for WordNet label generation. **[Previously missing from this list]**

## 13. Training / optimization infrastructure (Methods-only)

Methods-only references do not count against the main ~50 reference limit.

- **KEEP** · Paszke et al. (2019). *PyTorch: an imperative style, high-performance deep learning library.* **NeurIPS**. — Standard Methods citation.
- ~~REMOVE~~ · ~~Kingma & Ba (2015). *Adam: a method for stochastic optimization.* **ICLR**.~~ — Paper uses AdamW, not Adam. Cite Loshchilov & Hutter instead.
- **ESSENTIAL** · Loshchilov & Hutter (2019). *Decoupled weight decay regularization (AdamW).* **ICLR**. — Already cited.
- ~~REMOVE~~ · ~~Ioffe & Szegedy (2015). *Batch normalization.* **ICML**.~~ — BN is ubiquitous; no citation needed in Nature.
- ~~REMOVE~~ · ~~Ba, Kiros, Hinton (2016). *Layer normalization.* **arXiv**.~~ — Not used.
- **KEEP** · Pedregosa et al. (2011). *scikit-learn: machine learning in Python.* **JMLR**. — SRP from scikit-learn. Standard Methods credit.

---

## ~~Section removed: PCA / dimensionality reduction~~

~~Pearson (1901), Hotelling (1933), Jolliffe & Cadima (2016)~~ — PCA is universally understood. No Nature reviewer expects a citation for PCA itself.

## ~~Section removed: Low-data / data efficiency~~

~~Sorscher et al. (2022), Entezari et al. (2023)~~ — The data efficiency result (Fig. 4D) is a finding of this paper, not a literature claim that needs citation support.

---

## Summary

| Verdict | Count |
|---|---|
| ESSENTIAL | 23 |
| KEEP | 27 |
| REMOVE | 21 |
| **Total for Zotero import** | **50** |

---

## Notes for the verification pass

1. **DINOv3 citation fix:** `methods.md` cites "Oquab et al., 2024" for DINOv3 but that's the DINOv2 TMLR paper. DINOv3 is Siméoni et al. 2025 (arXiv:2508.10104). Both citations needed: Oquab for DINOv2, Siméoni for DINOv3.
2. **Schrimpf version fix:** `methods.md` currently cites the 2018 bioRxiv preprint. Should be updated to the 2020 Neuron publication (doi:10.1016/j.neuron.2020.07.040).
3. Several dataset papers (NSD, THINGS-data, TVSD) have both a preprint and a published version — use the **published** one; keep arXiv ID as a secondary `eprint` field per our BibTeX conventions.
4. Per the rules in `CLAUDE.md`, this file is a **candidate list only** — no Zotero imports until the user verifies and approves.
5. All **[NEW]** additions were identified via web search (April 2026) and need full field-by-field verification before import.
6. References tagged REMOVE are struck through but retained in this file for the record. They can be deleted entirely once the final bibliography is settled.

## New additions from literature search (2024–2026)

Papers added during the April 2026 literature search, grouped by relevance:

**Highest relevance (directly addresses the same question):**
- Elmoznino et al. (2025), Nature — coarse-grained abstraction alignment (user-requested)
- Badwal, Bergmann, Roth, Doeller & Hebart (2025), J Neurosci — scope of fine-grained info in ventral pathway
- Mahner, Muttenthaler, Güçlü & Hebart (2025), Nature Machine Intelligence — dimensions of human-DNN alignment

**High relevance (informs the broader argument):**
- Kazemian, Elmoznino & Bonner (2025), Nature Machine Intelligence — untrained CNNs are cortex-aligned
- Margalit et al. (2024), Neuron — TDANN: SSL yields better functional organization
- Elmoznino & Bonner (2024), PLoS Comp Bio — high dimensionality benefits cortex prediction
- Muttenthaler et al. (2023), NeurIPS — improving representations via human similarity judgments
- Storrs et al. (2021), J Cogn Neurosci — diverse DNNs all predict IT well

**Supporting (verifies existing citations / fills gaps):**
- Cichy et al. (2016), Scientific Reports — spatio-temporal hierarchical correspondence
- Mehrer et al. (2021), PNAS — ecologically motivated training data
- Fel et al. (2022), NeurIPS — harmonizing DNN and human recognition strategies
