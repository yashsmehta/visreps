Probing the granularity of human-machine alignment

Raj Magesh Gauthaman
Department of Cognitive Science
Johns Hopkins University
rgautha1@jh.edu

Yash Mehta
Department of Cognitive Science
Johns Hopkins University
ymehta3@jh.edu

Michael F. Bonner
Department of Cognitive Science
Johns Hopkins University
mfbonner@jh.edu

Abstract
Deep neural networks for object classification align closely with human visual
representations, a correspondence that has been attributed to fine-grained category
supervision. We investigate whether such granular supervision is necessary for robust brain-model alignment. Using a PCA-based method, we generate progressively
coarser ImageNet label sets (ranging from 2 to 64 categories) and retrain a standard
CNN (AlexNet) from scratch for each granularity, enabling controlled comparisons
against standard 1000-class training. Evaluations employ representational similarity analysis (RSA) on large-scale fMRI data (NSD, including out-of-distribution
stimuli) and behavioral data (THINGS). Our key findings include: (1) On behavioral data, models trained with minimal categories (e.g., 2 classes) achieve
surprisingly high alignment with human similarity judgments. (2) On fMRI data,
models trained with 32-64 categories match or outperform 1000-class models in
early visual cortex alignment and exhibit comparable performance in ventral areas,
with coarser models displaying advantages on OOD stimuli. (3) Coarse-trained
representations differ structurally from low-dimensional projections of fine-grained
models, suggesting the learning of novel visual features. Collectively, these findings
indicate that broader categorical distinctions are often sufficient — and sometimes
more effective — for capturing cognitively salient visual structure, especially in
early visual processing and OOD contexts. This work introduces classification
granularity as a new framework for probing visual representation alignment, laying
the groundwork for more biologically-aligned vision systems.

1

Introduction

Human visual processing represents a pinnacle of biological computation, enabling efficient navigation and interaction with complex environments. The quest to understand and replicate this capability
has driven significant advances in artificial intelligence, particularly through deep neural networks
(DNNs). A remarkable finding in recent years has been the emergence of representations within these
networks that show striking similarities to the human ventral visual stream—the brain pathway critical
for object recognition [27, 18, 24, 25]. This correspondence has established brain-model alignment
as both a benchmark for AI progress and a window into biological vision. The comparison of DNNs
with the primate visual system has employed methodologies like representational similarity analysis
(RSA) [19] and large-scale benchmarking efforts such as Brain-Score [24, 25] and the Algonauts
Project [5, 2]. Initially, these comparisons spurred a drive towards developing models with maximal
Preprint.

alignment through innovations in architecture, task, and training data, with the prevailing goal of
identifying the "best" neural network model of the human visual system.
Recent findings challenge the notion of a singular optimal neural network for modeling human
vision. Conwell et al. [7] demonstrated that diverse networks with varied architectures, tasks, and
training data can achieve comparable levels of brain similarity. Similarly, Elmoznino and Bonner
[9] found that model performance is more strongly associated with latent dimensionality than with
specific task objectives. This evolving understanding is further supported by Chen and Bonner [4]
and Hosseini et al. [15], who show that despite variations in model design or training objectives,
networks often learn core representational axes that align with neural responses to natural images.
This suggests that underlying representational structures, rather than architectural specifics, may be
the primary determinants of brain alignment. Recent evidence—the limited role of architecture, the
strong influence of training data, and the emergence of core representational axes—raises a critical
question: Is the development of human-like visual representations contingent on learning to solve
challenging fine-grained vision tasks (e.g., 1000-way classification on ImageNet)? Or could models
trained to carve up the space of natural images at a much coarser level still align with neural and
behavioral patterns?
This question is pertinent for several reasons. First, human object recognition operates across multiple
levels of abstraction, from general categories to specific exemplars, yet it remains unclear which
level best maps onto alignment metrics or these shared representational axes. Second, increasing
categorical granularity may not be the optimal strategy for achieving robust, human-like generalization
[10]. Finally, recent findings that self-supervised and language-guided models align with visual
cortex activity [11, 23] suggest that explicit fine-grained labels may not be the only route to brain-like
representations.
To investigate label granularity’s impact on human-machine alignment, we introduce a principled
approach to generating coarse-grained label sets for ImageNet. Rather than using pre-existing semantic hierarchies or external embeddings that might introduce confounding factors, we derive category
structures directly from the internal representations of a pre-trained vision model (AlexNet trained on
1000-way ImageNet classification). Using principal component analysis on these representations,
we recursively partition ImageNet classes to create hierarchical label sets with progressively fewer
categories (from 64 down to 2 coarse-classes), while keeping the 1.28 million training images constant
across conditions. For each granularity level, we train an AlexNet model from scratch specifically for
that K-way classification task, allowing us to isolate label granularity effects.
Our contributions are as follows:
• Coarse-grained training outperforms fine-grained models in human-machine alignment: We
demonstrate that models trained on broader categorical distinctions (32-64 classes) achieve
comparable or superior neural alignment to 1000-class models, particularly in early visual
areas, while also showing stronger behavioral alignment with human similarity judgments.
• Minimal supervision yields strong behavioral alignment: Surprisingly, training with just two
categories achieves the highest alignment with human behavioral judgments in the THINGS
dataset, suggesting that minimal categorical distinctions can effectively capture core aspects
of human visual similarity.
• Generalizable visual representations for out-of-distribution stimuli: Models trained on
coarser categories maintain higher neural alignment when tested on synthetic and unconventional visual inputs, indicating that broader distinctions may better capture generalizable
visual processing principles.
• A new lens for human-machine representational alignment: We introduce principled classification granularity as a promising lens for studying visual representations, diverging from
conventional architecture-centric approaches. To facilitate reproducible research, we release
an open-source library with modular experimentation tools and all model checkpoints.
Our findings suggest that learning broader visual distinctions may offer a more efficient path to
capturing cognitively salient structures, potentially leading to more robust visual AI systems that
align with principles of biological vision.
2

2

Related Work

While many foundational studies established the strong alignment between DNNs trained with standard fine-grained ImageNet labels and the primate ventral visual stream [16, 3, 6], the specific role
of this fine granularity, as opposed to coarser distinctions, has been less systematically explored.
Previous attempts to investigate label granularity have often relied on pre-defined semantic hierarchies
(e.g., WordNet) to define coarse and fine categories, or have focused on curriculum learning. For
instance, studies have explored curriculum learning strategies—such as training models sequentially
from coarse-to-fine labels or vice-versa [26, 14] — primarily with the goal of improving image
classification accuracy, generalization, or transfer learning performance, rather than directly optimizing for or extensively analyzing neural or behavioral alignment via RSA. While Wang and Cottrell
[26] showed coarse-to-fine pretraining improved transfer, and Hong et al. [14] identified "optimal
granularity" for task performance via fine-to-coarse transfer, direct RSA-based comparison to human
data was not their central focus. Similarly, work on multi-granular or hierarchical label supervision
[22, 21], where models might be trained simultaneously on, for example, basic and subordinate
categories, has primarily explored the structure of the learned feature spaces or classification benefits,
without systematically examining how varying label granularity impacts representational similarity
to human neural or behavioral data. These approaches, by relying on external semantic structures,
may also introduce biases that are distinct from the information purely derivable from visual input.
The study most directly comparable to ours is Ahn, Zelinsky, and Lupyan [1]. They systematically
manipulated label granularity based on semantic categories (superordinate, basic) and explored
curriculum orders (superordinate→basic, basic→superordinate) for models trained on ImageNet.
Using RSA, they compared the resulting network representations to human fMRI data (early visual
cortex, LOC) and behavioral similarity judgments, finding that a coarse-to-fine (superordinate→basic)
curriculum yielded the most human-like representations. Our work extends these prior efforts through
a representation-first approach using PCA on trained network features to define category boundaries,
avoiding reliance on external semantic taxonomies. We evaluate our models on more extensive
datasets (NSD for fMRI and THINGS for behavioral similarity) and demonstrate that coarse-trained
representations are not simply low-dimensional projections of fine-grained models but constitute
distinct learned structures.
It is also important to consider that strong brain alignment can be achieved through paradigms other
than fine-grained supervised object classification. Self-supervised learning (SSL) methods, which
learn representations by solving pretext tasks on unlabeled data, have been shown to produce features
that align well with the visual cortex, sometimes rivaling supervised approaches [17]. Similarly,
models pre-trained with vision-language objectives (e.g., CLIP [23]) also exhibit high neural similarity.
These findings collectively suggest that explicit, human-defined fine-grained object labels are not the
sole route to inducing brain-like representations.
Our research complements these alternative approaches. While SSL and language supervision explore
what else can lead to alignment (i.e., different types of objective functions or data modalities), our
work provides a systematic dissection of label granularity within the supervised object classification
framework. By keeping the supervision minimal yet systematically varying its informational content,
we aim to understand the lower bounds and optimal levels of explicit categorical information required
for robust human-model alignment.

3

Methods

This section details the procedures for generating coarse-grained label sets, training neural network
models, and the analytical techniques employed to compare their internal representations. The basic
training pipeline is illustrated in Figure 1.
3.1

PCA-based coarse-graining of ImageNet labels

To investigate the impact of label granularity on learned representations, we developed a method to
generate progressively coarser versions of the ImageNet-1k dataset [8]. The core idea is to create
hierarchical label sets (K = {2, 4, 8, 16, 32, 64} classes) directly from the representational geometry
of a network trained on the original 1000-class task, while maintaining the same set of 1.28 million
training images.
3

We began by extracting image representations from the fc2 layer of a pre-trained AlexNet [20],
trained on the 1000-way ImageNet classification task. This layer provides a feature space reflecting
fine-grained distinctions learned through the original labels. Additionally, a control analysis was
conducted using an untrained network, with those results provided in the Appendix.
Principal Component Analysis (PCA): PCA was performed on the fc2 representations of all 1.28
million ImageNet training images to identify the major axes of variance in the feature space.
Recursive median splits:
To form K = 2 classes, all images were projected onto the first principal component (PC1). A median
split on the projected values divided the dataset into two superclasses. The process was repeated for
each subsequent granularity level, using the next PC in the sequence:
• For K = 4, the two existing superclasses were projected onto PC2, with a median split
applied within each superclass.
• For K = 8, PC3 was used to further split the existing four classes.
• This process continued using PC4, PC5, and PC6 to generate K = 16, K = 32, and K = 64
classes, respectively.
This PCA-based coarse-graining approach offers several key advantages:
1. Consistency across granularity levels: The same set of 1.28 million training images is used
across all levels of granularity K, allowing for controlled comparisons.
2. Progressive hierarchy: The recursive splits generate a natural 2 → 4 → · · · → 64 class hierarchy
with balanced class distributions at each level.
3. Data-driven structure: Unlike human-defined hierarchies (e.g., WordNet) or external embeddings
(e.g., CLIP), this approach leverages learned feature variance from the original 1000-way model to
define class boundaries.
It is important to note that while the principal components are derived from the 1000-way model,
there is no information leakage, as each K-way model is independently re-trained from scratch with
randomly initialized weights.
3.2

Representational Similarity Analysis (RSA)

Representational Similarity Analysis (RSA) [19] quantifies the correspondence between representational spaces by comparing their representational similarity matrices (RSMs). RSA abstracts away
from specific data formats by evaluating the geometry of activations across different models and
between models and neural/behavioral data (see Sections 4 and 6 for NSD/THINGS evaluation).
For a given network layer (or brain region) and a set of N stimuli, activation patterns are extracted,
and an N × N RSM is constructed. Each entry RSMij quantifies the similarity between the
representations of stimuli i and j, calculated as the Pearson correlation (ρ) between the vectorized
activation patterns:
RSMij =

cov(acti , actj )
= ρ(acti , actj )
σacti σactj

(1)

The diagonal elements are either ignored or set to one.
To assess similarity between two representational geometries (e.g., two model layers or two datasets),
the off-diagonal elements of the RSMs are compared using Spearman’s rank correlation (ρs ). A
higher ρs indicates a closer alignment in the representational structures.
3.3 Characterizing representation differences: learned vs. PCA reconstructed features
A key objective in this analysis is to determine whether representations learned through coarse-grained
classification tasks differ qualitatively from those obtained by reconstructing fine-grained model
features using a subset of principal components (PCs). PCA reconstruction involves extracting the
top M PCs from a model trained on the 1000-class task and reconstructing feature vectors in the
reduced subspace defined by these M PCs. This serves as a controlled baseline for comparison with
K-way models trained directly on M -dimensional label sets.
4

Create synthetic class labels using the PCs of the
Create synthetic class labels using the PCs of the
fc21latent
space of an ImageNet-trained AlexNet.

1

2

fc2 latent space of an ImageNet-trained AlexNet.

1

A

Create synthetic class labels using the PCs of the
fc2 latent space of an ImageNet-trained AlexNet.

PC 1

PC 1

PC 1

PC 1

2

Train models on these
synthetic class labels.

PC 1

B

PC 1

ImageNet

Train models on these
Train models on these
class labels.
2 synthetic
synthetic class labels.

k-way
classiﬁcation

ImageNet
ImageNet

k-way
k-way
classiﬁcation
classiﬁcation

PC 2

1000 classes

2 classes

1000 CLS

4 classes

...

2 CLS

4 CLS

1000
classes
2 classes
4 classes
Evaluate model
similarity
using representational
similarity analysis (RSA).

3

C 1000 classes

2 classes

model features

3

3

PC PC
2 2

AlexNet

4 classes

...

...

stimuli

CNN

AlexNet

AlexNet

D

RSMX
RSM
Evaluate
model similarity using representational
similarity
analysis (RSA).
Y
i

stimuli

j

X

stimuli

Evaluate model similarity using representational similarity analysis (RSA).
model features

model featuresPearson r(Xi, Xj)
stimuli

i

representational
i matrixj (RSM)
similarity

X and Y

RSA correlation
Spearman r(RSMX, RSMY)

stimuli

layer
activations

stimuli

compare two representations
stimuli

RSMX

RSMX

RSMY

RSMY

stimuli

stimuli

j
X
compare
two representations
Pearsoncoarse-graining,
r(Xi, Xj)
Figure 1: Overview of methods: PCA-based
RSA. (A)
Coarse-graining:
Original
X andalong
Y
X
ImageNet labels are replaced with PCA-derived categories, progressively
splitting
PCs to
compare two representations
Pearson
i, Xj)
generate 2, 4, 8, 16, 32,
and 64 classes.
(B)r(X
Independent
network training at each
granularity
without
layer
representational
RSA
correlation
X and Y
curriculum learning.
(C) RSA: Representational
matrices (RSMs)
capturer(RSM
pairwise
correlaactivations
similarity similarity
matrix (RSM)
Spearman
X, RSMY)
tions of modellayer
and brain activations, quantifying
representational shifts across
levels. (D)
representational
RSAtraining
correlation
PCA reconstruction
control: RSA scores
compare
against PCA-reconstructed
activations
similarity
matrixK-way
(RSM)trained models
Spearman
r(RSMX, RSMY)
1000-way models to assess whether coarse training yields distinct representations.

RSA for K-way trained models: RSMs were computed from each layer of AlexNet models trained
from scratch on the K-way classification tasks (e.g., 4-way classification corresponding to M = 2
PCs).
RSA for PCA reconstructed 1000-way models: Activations from corresponding layers of a 1000way AlexNet model were extracted and projected onto the top M PCs. The feature vectors were then
reconstructed by reversing the projection, producing M -dimensional representations used to compute
RSMs.
Comparative analysis: Spearman correlations were calculated between RSMs derived from K-way
trained models and those derived from PCA-reconstructed 1000-way models across all network
layers.
Benchmark comparisons:
K-way model intrinsic similarity: RSA scores between independently trained instances of the
K-way model, providing a reference for the stability of coarse-trained representations.
PCA reconstructed 1000-way intrinsic similarity: RSA scores between independently trained
1000-way models after reconstructing feature vectors in the M -dimensional subspace, serving as a
baseline for how stable the PCA-reconstructed features are.
A substantial divergence in RSA scores between the K-way learned representations and PCAreconstructed 1000-way features would suggest that the coarse-grained training leads to qualitatively
distinct representations rather than mere dimensionality-reduced versions of fine-grained models.
Results are presented in Figure 1.

4

NEURAL Evaluation: Model-brain alignment on naturalistic scene viewing

We assess the alignment between model representations and neural responses during naturalistic
scene viewing, focusing on coarse-trained models and PCA-reconstructed features to isolate the
contributions of categorical granularity to brain alignment.
5

B

Natural Scenes Dataset
natural scenes

functional MRI

Early Visual Stream

C

Early Visual Stream

73,000 images

A

8 subjects

early visual
stream

D

Ventral Visual Stream

E

Ventral Visual Stream

stimuli

voxels

ventral visual
stream

−3

−2

−1

0

1

activations (z-scored)

2

3

Figure 2: (A) Experimental paradigm: Participants viewed natural scenes while fMRI data were
recorded from the early visual stream (low-level features) and ventral visual stream (object-level
processing). (B, D) Brain similarity scores across K-way training levels, with darker colors indicating
finer granularity. (B) Early visual stream, (D) Ventral visual stream. (C, E) Reconstruction analysis:
PCA projection of 1000-way model activations onto top PCs fails to reach the brain similarity scores
achieved by direct coarse training (e.g., 32-way classification), highlighting the distinctiveness of
coarse-trained representations.
4.1

Natural Scenes Dataset (NSD) primer

The NSD is the largest human fMRI dataset in terms of the number of stimulus presentations
per participant, with each subject exposed to over 9,000 unique images and 1,000 shared images
across 30-40 sessions, totaling over 22,500 trials per subject [2]. This level of stimulus sampling
is unprecedented, allowing for highly granular analysis of neural responses to natural scenes. Its
extensive data coverage, coupled with 7T resolution and multimodal imaging (including anatomical,
resting-state, retinotopic mapping, and diffusion scans), has made it a benchmark dataset for studies
on visual representation, neural alignment, and computational model evaluation, contributing to its
widespread popularity and adoption in neuroscience and AI research.
4.2

Insights

Coarse 32-way training: enhanced early visual alignment
For the early visual stream, 32-way classification achieves substantially higher representational
alignment compared to the 1000-way model. Even a 4-way model reaches alignment comparable to
the 1000-way model, but increasing the granularity to 32 further enhances alignment. Beyond 32
classes, alignment slightly declines, suggesting a sweet spot for capturing early visual features.
Coarse 64-way training: ventral stream parity with fine-grained models
In the ventral visual stream, 32 and 64-way classification models maintain strong alignment, approaching levels seen in 1000-way models. The trend is particularly notable when contrasted with untrained
models, which serve as a low-performing baseline. Alignment steadily improves as granularity
increases from 2 to 64, highlighting the ventral stream’s sensitivity to intermediate-level category
distinctions.
4.3

PCA-reconstruction control

This control experiment assesses whether the observed alignment gains from coarse training are
genuinely due to learning distinct representations or simply a byproduct of dimensionality reduction.
Reconstruction using top PCs has been observed to increase brain alignment, likely by filtering out
high-rank, noisy components. Here, we project model activations onto the top N PCs and reconstruct
them to verify that the alignment gains from coarse training are not merely due to noise reduction.
6

Ventral Visual Stream

NSD - Synthetic
Dataset

NSD Dataset

Early Visual Stream

Figure 3: RSA analysis across datasets and brain regions. RSA scores across conv1–conv5, fc1,
fc2 for NSD (top) and NSD Synthetic (bottom). Left: Early visual stream — Coarse training (16, 32,
64-way) yields higher scores in conv layers, particularly under OOD stimuli. Right: Ventral visual
stream — Finer granularity (32, 64-way) boosts alignment in NSD; alignment remains stable across
levels in NSD Synthetic.

Early visual stream: The 32-way coarse training model consistently outperforms all PC-based
reconstructions from the 1000-way model, even when using the top 1–20 PCs. The 1000-way model,
even after dimensionality reduction, cannot achieve close to the same level of brain alignment as
the 32-way model. This indicates that the 32-way representations are not merely lower-dimensional
projections but instead capture more behaviorally relevant visual features.
Ventral visual stream: In the ventral stream, alignment follows the expected pattern: it increases
with the inclusion of the first 1–10 PCs but then declines as higher-rank PCs (11–20) are added, likely
due to the inclusion of noise. This pattern is consistent with previously observed trends and serves as
a sanity check to confirm expected behavior.

5

NEURAL Evaluation II: Model-brain alignment on OOD stimuli

This section explores how model-brain alignment varies when evaluated on out-of-distribution (OOD)
stimuli, using the NSD-Synthetic dataset. The goal is to assess whether training on fine-grained
categories effectively generalizes to unconventional visual inputs.

5.1

Natural Scenes Dataset-Synthetic primer

NSD-Synthetic is a smaller dataset consisting of 284 unconventional stimuli, such as noise patterns,
colors, words, and spirals, presented to the same eight participants from the NSD dataset under
identical recording conditions [12]. Unlike typical visual datasets, NSD-Synthetic includes stimuli
that are highly dissimilar to ImageNet-style natural scenes, making it a compelling test for assessing
model robustness to OOD stimuli.
7

A

THINGS (behavioral)

B

THINGS (behavioral)

C

THINGS (behavioral)

Figure 4: Behavioral alignment via RSA. (A) RSA scores across conv1–fc2 for K-way training
(2–1000 classes). Two-way training maximizes alignment, with no gains from finer granularity. (B)
RSA scores by granularity: Coarse (2-way) training consistently outperforms finer classes, indicating
limited behavioral relevance of fine-grained distinctions. (C) Reconstruction analysis: RSA scores for
PCA-reconstructed 1000-way activations (1–20 PCs) remain below two-way training, highlighting
the distinctiveness of coarse-trained representations.

5.2

Insights

The layer-wise plots illustrate how model-brain alignment varies across network layers for both
NSD (natural scenes) and NSD-Synthetic (out-of-distribution stimuli), enabling direct comparison of
alignment trends between in-distribution and OOD images.
Coarse training: enhanced alignment on OOD stimuli
Coarse-grained training shows stronger alignment to the early visual stream, particularly in the
convolutional layers, for both natural scenes and OOD stimuli. The effect is most pronounced in
the early visual stream, where even very coarse models (e.g., 8-way) achieve notable alignment. On
average across layers, OOD stimuli maintain higher brain alignment under coarse training, suggesting
that broader categorical distinctions capture essential visual features that generalize effectively.
Classification granularity: limited impact on OOD brain score performance
For natural scenes, increasing the number of training categories substantially improves brain alignment, especially in the ventral visual stream. However, with OOD stimuli, the impact of training
granularity is less pronounced. While trained models still outperform untrained models in OOD
settings, the variance in alignment across different coarseness levels is minimal, indicating that finer
category distinctions do not substantially enhance brain alignment in OOD conditions.

6

BEHAVIORAL Evaluation: model-brain alignment on a visual odd-one-out
task

This section explores how well model representations align with human perceptual similarity as
assessed using a visual odd-one-out task based on the THINGS dataset.
6.1

THINGS dataset

The THINGS dataset is a large-scale benchmark for studying object representations in human
perception and cognition, comprising 1,854 diverse object concepts with 26,107 high-quality images
[13]. It has become a widely used resource for evaluating behavioral alignment because it captures
how humans perceive similarity across a broad range of objects. The key feature of the dataset is a
66-dimensional embedding derived from human similarity judgments. Participants rated how similar
different objects felt to them, and these ratings were then distilled into 66 core dimensions that capture
perceptual axes like size, shape, and function. These embeddings provide a structured way to quantify
behavioral similarity across objects, making it possible to compare how well computational models
align with human perception in a standardized way. As a result, THINGS has become a standard
resource in the community for benchmarking behavioral alignment in visual representation studies.
8

6.1.1

Insights

For all PC-based analyses, fc2 (penultimate layer) is used as it shows the highest alignment, except for
the 1000-way model, where fc1 has the maximum alignment. We also evaluate multiple correlation
metrics (Spearman, Kendall, Pearson) for computing RSMs, and the observed trends remain consistent
across all metrics.
Maximally coarse training (2-way) achieves the highest behavioral similarity:
Training on just two-way classification yields the highest behavioral alignment, substantially surpassing even the 1000-way model. This effect is further reinforced by the PC reconstruction analysis,
which shows that no lower-rank reconstruction from the 1000-way model comes close to the behavioral similarity achieved by the 2-way training.
Behavioral alignment remains invariant to classification granularity:
Whether trained on four-way or 64-way classification, the underlying learned representations differ
substantially, but behavioral alignment remains largely unchanged. The effect of granularity is
negligible, indicating that finer distinctions do not meaningfully improve alignment. However, the
substantial gap between untrained and trained models suggests that some level of training is necessary
to achieve behavioral alignment.

7

Discussion and Limitations

Our findings reveal that training deep neural networks on coarser class labels yields substantially
higher alignment with early visual stream representations compared to fine-grained classification
tasks. This effect is consistently observed across both natural and synthetic stimuli, underscoring the
robustness of coarse-grained training in capturing neural representations that align more closely with
early-stage visual processing. Furthermore, we demonstrate that two-way training — a minimalistic
classification task — achieves unexpectedly high behavioral alignment with human vision. These
results open up a new line of inquiry into how the granularity of classification tasks influences
neural alignment, positioning coarse-grained training as an informative framework for probing
representational principles in both artificial and biological systems. This approach diverges from
conventional model comparison paradigms, which predominantly focus on achieving high neural
similarity via complex architectures or fine-grained training schemes. Instead, our work suggests that
simplification in classification granularity can serve as a powerful lens through which to interrogate the
structure of learned representations. Our findings also prompt reconsideration of existing benchmarks
in neural alignment research. While much of the current focus is on optimizing deep learning models
through more complex architectures, our work demonstrates that altering task structure — specifically
classification granularity — may offer an alternative and potentially more insightful avenue for
examining representational correspondence. Rather than prioritizing model complexity, we advocate
for leveraging controlled experimental setups to isolate key representational principles underlying
visual processing.
However, several limitations warrant acknowledgment. While our findings demonstrate intriguing alignment patterns across different granularities, a deeper mechanistic understanding of why
certain coarse-grained representations yield high similarity scores with brain activity remains an
important area for future exploration. Building on these findings, several exciting avenues for future
research emerge. Investigating curriculum learning effects—particularly how alignment patterns
evolve through coarse-to-fine granularity transitions—could reveal fundamental principles of visual
processing in both biological and artificial systems. Recent work on universal dimensions of visual
representations presents an opportunity to compare our discovered coarse-grained structures with
these proposed fundamental features [4]. Perhaps most promising is the extension to encoding models
beyond representational similarity analysis, which would enable direct prediction of brain activity
and creation of whole-brain maps showing optimal granularity levels for each voxel. This could
reveal brain-wide organizational principles and validate trends suggested by our initial RSA findings.
Our work points to classification granularity as a potent tool for dissecting neural-artificial alignment,
bridging insights between computational models and visual processing in the brain.

9

References
[1]

[2]

[3]

[4]
[5]

[6]

[7]

[8]

[9]

[10]

[11]

[12]

[13]

[14]
[15]

[16]

Seoyoung Ahn, Gregory J. Zelinsky, and Gary Lupyan. “Use of superordinate labels yields
more robust and human-like visual representations in convolutional neural networks”. In:
Journal of Vision 21.13 (Dec. 2021), p. 13. ISSN: 1534-7362. DOI: 10.1167/jov.21.13.13.
URL: http://dx.doi.org/10.1167/jov.21.13.13.
Emily J. Allen et al. “A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial
intelligence”. In: Nature Neuroscience 25.1 (Dec. 2021), pp. 116–126. ISSN: 1546-1726. DOI:
10.1038/s41593-021-00962-x. URL: http://dx.doi.org/10.1038/s41593-02100962-x.
Charles F. Cadieu et al. “Deep Neural Networks Rival the Representation of Primate IT Cortex
for Core Visual Object Recognition”. In: PLoS Computational Biology 10.12 (Dec. 2014). Ed.
by Matthias Bethge, e1003963. ISSN: 1553-7358. DOI: 10.1371/journal.pcbi.1003963.
URL: http://dx.doi.org/10.1371/journal.pcbi.1003963.
Zirui Chen and Michael F. Bonner. Universal dimensions of visual representation. 2024. DOI:
10.48550/ARXIV.2408.12804. URL: https://arxiv.org/abs/2408.12804.
R. M. Cichy et al. The Algonauts Project 2021 Challenge: How the Human Brain Makes
Sense of a World in Motion. 2021. DOI: 10 . 48550 / ARXIV . 2104 . 13714. URL: https :
//arxiv.org/abs/2104.13714.
Radoslaw Martin Cichy, Dimitrios Pantazis, and Aude Oliva. “Similarity-Based Fusion of
MEG and fMRI Reveals Spatio-Temporal Dynamics in Human Cortex During Visual Object
Recognition”. In: Cerebral Cortex 26.8 (May 2016), pp. 3563–3579. ISSN: 1460-2199. DOI:
10.1093/cercor/bhw135. URL: http://dx.doi.org/10.1093/cercor/bhw135.
Colin Conwell et al. “A large-scale examination of inductive biases shaping high-level visual
representation in brains and machines”. In: Nature Communications 15.1 (Oct. 2024). ISSN:
2041-1723. DOI: 10.1038/s41467-024-53147-y. URL: http://dx.doi.org/10.1038/
s41467-024-53147-y.
Jia Deng et al. “ImageNet: A large-scale hierarchical image database”. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. IEEE. 2009, pp. 248–255. DOI: 10.1109/
CVPR.2009.5206848. URL: https://ieeexplore.ieee.org/document/5206848.
Eric Elmoznino and Michael F. Bonner. “High-performing neural network models of visual
cortex benefit from high latent dimensionality”. In: PLOS Computational Biology 20.1 (Jan.
2024). Ed. by Drew Linsley, e1011792. ISSN: 1553-7358. DOI: 10.1371/journal.pcbi.
1011792. URL: http://dx.doi.org/10.1371/journal.pcbi.1011792.
Robert Geirhos et al. “ImageNet-trained CNNs are biased towards texture; increasing shape
bias improves accuracy and robustness”. In: International Conference on Learning Representations (ICLR). 2019. URL: https://arxiv.org/abs/1811.12231.
Amir Ghorbani et al. “Towards aligning artificial and biological vision systems with selfsupervised learning”. In: arXiv preprint arXiv:2106.13884 (2021). URL: https://arxiv.
org/abs/2106.13884.
Alessandro T. Gifford et al. A 7T fMRI dataset of synthetic images for out-of-distribution
modeling of vision. 2025. DOI: 10.48550/ARXIV.2503.06286. URL: https://arxiv.
org/abs/2503.06286.
Martin N Hebart et al. “THINGS-data, a multimodal collection of large-scale datasets for
investigating object representations in human brain and behavior”. In: eLife 12 (Feb. 2023).
ISSN : 2050-084X. DOI : 10.7554/elife.82580. URL : http://dx.doi.org/10.7554/
eLife.82580.
Guan Zhe Hong et al. Towards Understanding the Effect of Pretraining Label Granularity. 2023.
DOI : 10.48550/ARXIV.2303.16887. URL : https://arxiv.org/abs/2303.16887.
Eghbal Hosseini et al. “Universality of representation in biological and artificial neural networks”. In: (Dec. 2024). DOI: 10.1101/2024.12.26.629294. URL: http://dx.doi.org/
10.1101/2024.12.26.629294.
Seyed-Mahdi Khaligh-Razavi and Nikolaus Kriegeskorte. “Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation”. In: PLoS Computational Biology 10.11
(Nov. 2014). Ed. by Jörn Diedrichsen, e1003915. ISSN: 1553-7358. DOI: 10.1371/journal.
pcbi.1003915. URL: http://dx.doi.org/10.1371/journal.pcbi.1003915.

10

[17] Talia Konkle and George A. Alvarez. “A self-supervised domain-general learning framework
for human ventral stream representation”. In: Nature Communications 13.1 (Jan. 2022). ISSN:
2041-1723. DOI: 10.1038/s41467-022-28091-4. URL: http://dx.doi.org/10.1038/
s41467-022-28091-4.
[18] Nikolaus Kriegeskorte. “Deep Neural Networks: A New Framework for Modeling Biological
Vision and Brain Information Processing”. In: Annual Review of Vision Science 1.1 (Nov.
2015), pp. 417–446. ISSN: 2374-4650. DOI: 10.1146/annurev-vision-082114-035447.
URL: http://dx.doi.org/10.1146/annurev-vision-082114-035447.
[19] Nikolaus Kriegeskorte. “Representational similarity analysis – connecting the branches of
systems neuroscience”. In: Frontiers in Systems Neuroscience (2008). ISSN: 1662-5137. DOI:
10.3389/neuro.06.004.2008. URL: http://dx.doi.org/10.3389/neuro.06.004.
2008.
[20] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. “ImageNet classification with deep
convolutional neural networks”. In: Advances in Neural Information Processing Systems.
Vol. 25. Curran Associates, Inc., 2012, pp. 1097–1105. URL: https : / / proceedings .
neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b- Abstract.
html.
[21] Yinhua Li, Shouhong Wan, and Peiquan Jin. “Label hierarchy constraint network for finegrained classification”. In: Thirteenth International Conference on Digital Image Processing
(ICDIP 2021). Ed. by Xudong Jiang and Hiroshi Fujita. SPIE, June 2021, p. 22. DOI: 10.
1117/12.2599588. URL: http://dx.doi.org/10.1117/12.2599588.
[22] Joshua C. Peterson et al. Learning Hierarchical Visual Representations in Deep Neural
Networks Using Hierarchical Linguistic Labels. 2018. DOI: 10.48550/ARXIV.1805.07647.
URL: https://arxiv.org/abs/1805.07647.
[23] Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision”. In: Proceedings of the 38th International Conference on Machine Learning (ICML).
2021. URL: https://arxiv.org/abs/2103.00020.
[24] Martin Schrimpf et al. “Brain-Score: Which Artificial Neural Network for Object Recognition
is most Brain-Like?” In: (Sept. 2018). DOI: 10.1101/407007. URL: http://dx.doi.org/
10.1101/407007.
[25] Martin Schrimpf et al. “Integrative Benchmarking to Advance Neurally Mechanistic Models
of Human Intelligence”. In: Neuron 108.3 (Nov. 2020), pp. 413–423. ISSN: 0896-6273. DOI:
10.1016/j.neuron.2020.07.040. URL: http://dx.doi.org/10.1016/j.neuron.
2020.07.040.
[26] Panqu Wang and Garrison W. Cottrell. Basic Level Categorization Facilitates Visual Object
Recognition. 2015. DOI: 10.48550/ARXIV.1511.04103. URL: https://arxiv.org/abs/
1511.04103.
[27] Daniel L. K. Yamins et al. “Performance-optimized hierarchical models predict neural responses in higher visual cortex”. In: Proceedings of the National Academy of Sciences 111.23
(May 2014), pp. 8619–8624. ISSN: 1091-6490. DOI: 10.1073/pnas.1403112111. URL:
http://dx.doi.org/10.1073/pnas.1403112111.

11

NeurIPS Paper Checklist
1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: yes, it is in line with the results we show in Figure 3, 4, and 5.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: Please see section: Discussion and Limitations.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [NA]
12

Justification: There are no theoretical results in this manuscript that require a proof.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: yes, in addition, we also release all of the model weights and open-source
the GitHub repository for running all of these experiments and reproducing each of these
figures.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code
13

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: Yes, please check out the GitHub repository and all of the model checkpoints
along with training across various seeds that we have released as well.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: Yes
Justification: Yes, we aim to disclose everything in case anything is mistakenly missed
out. Then we have an extensive report of the hyperparameters and the training setup in the
appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: Yes
Justification: All results are reported on standard benchmarks, and that also the networks
are trained on multiple seeds from scratch, and answers are averaged over multiple subjects
as well (for NSD).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
14

• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: No
Justification: We do not provide the compute resources requirements in the main paper
because we felt it was tangential to the story of the paper. However, we do provide this in
the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
10. Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: Yes
Justification: There is not a specific broad societal impact of this work.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
15

• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: The models trained in this paper do not have a high risk of misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: Yes
Justification: We use open source datasets, which have been cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
16

• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification:
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: This paper does not involve crowdsourcing or research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15. Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: This paper does not involve crowdsourcing or research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
17

• We recognize that the procedures for this may vary substantially between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16. Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA]
Justification: LLM use is not an integral part of this research work or direction.
Guidelines:
• The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.

18

