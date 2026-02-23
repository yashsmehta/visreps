# Methods

## 1. PCA-based coarse-graining of ImageNet labels

A central challenge in studying how label granularity shapes learned representations is generating coarse label sets that vary supervision systematically without introducing confounding factors. Existing approaches have relied on pre-defined semantic hierarchies (e.g., WordNet superordinate categories) or external embeddings (e.g., language models), but these inject human-defined taxonomic biases that conflate label structure with semantic content. We instead derive category boundaries directly from the representational geometry of vision models, ensuring that the resulting labels reflect statistical regularities in visual feature space rather than linguistic or taxonomic conventions.

### Coarse label generation

We extract image representations from the entire ImageNet-1K training set (1,261,406 images) using a pretrained source model, then apply principal component analysis (PCA) to identify the dominant axes of variance in the resulting feature space. These principal components serve as a natural basis for recursively partitioning images into progressively coarser categories.

Concretely, all 1,261,406 feature vectors are mean-centred and projected onto the top 6 principal components. For each PC, a binary indicator is computed using a global median threshold: images whose projection falls above the median are assigned to one group, and those below to the other. To generate $K = 2^n$ classes, the binary indicators for the first $n$ PCs are concatenated and interpreted as a binary integer, yielding class assignments from 0 to $K - 1$. This produces a strictly nested hierarchy: the 2-class partition is a coarsening of the 4-class partition, which is a coarsening of the 8-class partition, and so on up to 64 classes (using 6 PCs). The resulting class sizes are approximately balanced (e.g., for the 32-class AlexNet-derived partition: minimum 15,923, maximum 76,584, mean 39,419 images per class).

### Source models for label generation

To ensure that our findings are not contingent on the particular representational geometry of a single source model, we generate coarse labels from four pretrained architectures spanning distinct training paradigms:

| Source model | Architecture | Training objective | Feature layer | Dimensionality |
|---|---|---|---|---|
| AlexNet | CNN (supervised) | ImageNet-1K classification | FC2 | 4,096 |
| CLIP | ViT-L/14 | Contrastive language-image pretraining | Image encoder output | 768 |
| DINOv2 | ViT-L/16 | Self-supervised self-distillation | CLS token | 1,024 |
| ViT | ViT-L/16 (supervised) | ImageNet-1K classification | CLS token | 1,024 |

All features are L2-normalised before PCA. The use of multiple source models — spanning supervised, self-supervised, and multimodal training — allows us to test whether the relationship between granularity and brain alignment is robust to the representational prior used to define the coarse labels.

### No information leakage

Although the principal components are derived from pretrained source models, there is no information leakage into the downstream analysis. Every model evaluated in this study is trained entirely from scratch with randomly initialised weights. The PCA-based labels define only the classification targets; the network architecture, optimisation, and all model parameters are independent of the source model. This is a critical distinction from approaches that fine-tune or adapt pretrained representations, and it ensures that any alignment differences we observe are attributable to the granularity of the learning objective, not to transferred features.

---

## 2. Model training

### Training from scratch on full ImageNet

A key feature of our experimental design is that every model is trained from scratch on the complete ImageNet training set (~1.26 million images). This stands in contrast to the common practice in brain-model alignment studies of evaluating off-the-shelf pretrained models from standard libraries (e.g., torchvision). While convenient, the off-the-shelf approach confounds the effects of architecture, training recipe, and label structure, since each pretrained model typically reflects a unique combination of these factors. By training all models ourselves under identical conditions — varying only the granularity of the classification labels — we achieve strict experimental control over the variable of interest.

In total, we train 87 models: for each of the four PCA source models (AlexNet, CLIP, DINO, ViT), we train at six coarseness levels ($K \in \{2, 4, 8, 16, 32, 64\}$), each with 3 independent random seeds, yielding 72 coarse-trained models. We additionally train 3 baseline models at the standard 1000-class granularity (one per seed) and 3 untrained (randomly initialised) models that serve as a lower bound. Every trained model sees the same ~1.26 million images for the same number of epochs; only the label mapping changes.

### Architecture

All experiments use a custom CNN following the AlexNet blueprint (Krizhevsky et al., 2012), modified with modern training practices. The network comprises five convolutional layers and two fully connected hidden layers, with batch normalisation after every convolutional and fully connected layer (except the final classifier), ReLU activations, max pooling after layers 1, 2, and 5, and dropout ($p = 0.3$) before each fully connected layer. Adaptive average pooling reduces feature maps to $3 \times 3$ spatial dimensions before the classifier. The final output layer has $K$ units, where $K$ matches the granularity condition. He (Kaiming) initialisation is used throughout.

| Layer | Operation | Channels/Units | Kernel | Stride | Padding | Spatial output |
|---|---|---|---|---|---|---|
| conv1 | Conv2d + BN + ReLU + MaxPool | 3 $\to$ 96 | 11 | 4 | 2 | 27 $\times$ 27 |
| conv2 | Conv2d + BN + ReLU + MaxPool | 96 $\to$ 256 | 5 | 1 | 2 | 13 $\times$ 13 |
| conv3 | Conv2d + BN + ReLU | 256 $\to$ 384 | 3 | 1 | 1 | 13 $\times$ 13 |
| conv4 | Conv2d + BN + ReLU | 384 $\to$ 384 | 3 | 1 | 1 | 13 $\times$ 13 |
| conv5 | Conv2d + BN + ReLU + MaxPool | 384 $\to$ 256 | 3 | 1 | 1 | 6 $\times$ 6 |
| pool | AdaptiveAvgPool2d | — | — | — | — | 3 $\times$ 3 |
| fc1 | Linear + BN + ReLU | 2,304 $\to$ 4,096 | — | — | — | — |
| fc2 | Linear + BN + ReLU | 4,096 $\to$ 4,096 | — | — | — | — |
| output | Linear | 4,096 $\to$ $K$ | — | — | — | — |

The total parameter count is approximately 34 million for 1000-class models and 30 million for 32-class models; the only architectural difference across conditions is the number of output units. We deliberately choose a relatively simple architecture (AlexNet-class) to ensure that observed effects reflect label granularity rather than the representational capacity of the network. AlexNet-class models remain among the most widely used architectures in computational neuroscience benchmarking, making our results directly comparable to prior work.

### Optimisation

All models are trained for 20 epochs using AdamW with a learning rate of $5 \times 10^{-4}$, weight decay of $10^{-3}$ (applied to weight matrices only; biases and batch normalisation parameters are excluded), and gradient clipping at a maximum norm of 1.0. We use cross-entropy loss with label smoothing ($\varepsilon = 0.1$) and a batch size of 32. The learning rate follows a cosine annealing schedule with linear warmup: the learning rate increases linearly from $0.25 \times \text{lr}$ to $\text{lr}$ over the first 2 epochs, then decays following a cosine schedule to $0.05 \times \text{lr}$ over the remaining 18 epochs.

Training images are resized to 256 pixels along the shorter edge, centre-cropped to $224 \times 224$, augmented with random horizontal flips ($p = 0.5$) and random rotations ($\pm 10°$), and normalised with ImageNet channel statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). Evaluation images undergo the same resize and centre-crop without augmentation.

### Reproducibility and seeds

Each granularity condition is trained with 3 independent random seeds (1, 2, 3). Deterministic training is enforced (`torch.manual_seed(seed)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`). The ImageNet train/test split uses a fixed generator seed (42) with an 80/20 ratio, ensuring identical train and test images across all conditions and seeds.

---

## 3. Activation extraction and dimensionality reduction

### Feature extraction via forward hooks

To compare model representations with neural and behavioural data, we extract internal activations from every layer of each trained model. Activations are captured via forward hooks attached to each convolutional and fully connected layer, recording the output of both the raw linear transformation (pre-activation) and the subsequent batch normalisation plus ReLU (post-activation). This yields 14 candidate extraction points per model (7 layers $\times$ 2 activation stages), providing fine-grained coverage of the representational hierarchy.

### Sparse random projection

Extracting full-resolution activations from all 14 candidate layers simultaneously would exceed available GPU memory for datasets of the scale we evaluate (thousands of high-resolution images $\times$ up to 64,896-dimensional feature maps per layer). We address this using sparse random projection (SRP), a dimensionality reduction technique grounded in the Johnson–Lindenstrauss lemma: distances and inner products in high-dimensional space are approximately preserved when projected to a lower-dimensional subspace via a random linear map.

For each layer, we project activations to $k = \min(4096, D)$ dimensions, where $D$ is the native flattened dimensionality. The projection uses scikit-learn's `SparseRandomProjection` with automatic density selection (approximately $1/\sqrt{D}$). The sparse projection matrix is converted to a PyTorch sparse tensor on GPU and applied on-the-fly during the forward pass via sparse matrix multiplication, enabling efficient extraction of all 14 layers in a single pass. Fitted projection matrices are cached to disk for reuse across evaluation runs.

Crucially, SRP serves different roles in our two alignment analyses. For representational similarity analysis (RSA), SRP is used only during layer selection to identify the most brain-aligned layer; the selected layer's activations are then re-extracted at full resolution without SRP for the final test evaluation, ensuring that test-phase RDMs are computed from exact, unprojected activations. For encoding score analysis, SRP-projected activations are used throughout, as the ridge regression operates on these projected features and the Johnson–Lindenstrauss guarantee ensures that the linear mapping quality is approximately preserved.

---

## 4. Evaluation datasets

We evaluate brain-model alignment across four datasets spanning three measurement modalities — functional MRI in humans, multi-unit electrophysiology in macaques, and behavioural similarity judgments — providing converging evidence across species and levels of neural organisation.

### 4.1 Natural Scenes Dataset (NSD)

The Natural Scenes Dataset (Allen et al., 2022) is the largest publicly available single-subject human fMRI dataset, with 8 participants each viewing over 9,000 unique natural scene photographs across 30–40 scanning sessions at 7T field strength. Functional data were preprocessed using the `fithrf_GLMdenoise_RR` GLM pipeline, which combines hemodynamic response function fitting, GLMdenoise, and ridge-regularised denoising to produce z-scored single-trial beta weights per voxel. Responses to repeated presentations of the same stimulus were averaged.

For each subject, stimuli were split into a subject-specific training set (~9,000 unique images) and a shared test set (~1,000 images viewed by all 8 subjects). Using the shared test set ensures that test-phase representational dissimilarity matrices (RDMs) are computed over identical stimuli across subjects, enabling meaningful cross-subject comparisons. We analyse two broad cortical streams (early visual stream and ventral visual stream) and six individual regions of interest (V1, V2, V3, hV4, FFA, PPA), spanning the full hierarchy from low-level retinotopic cortex to high-level category-selective areas. Stimulus images are loaded lazily from HDF5 to avoid loading the full ~36 GB image set into memory.

### 4.2 NSD-Synthetic (out-of-distribution evaluation)

To test whether coarse-grained training yields representations that generalise beyond the distribution of natural images, we evaluate on NSD-Synthetic (Gifford et al., 2025), a companion dataset comprising 220 unconventional stimuli — including noise patterns, colour fields, text, and spirals — presented to the same 8 NSD participants under identical scanning conditions. These stimuli are deliberately dissimilar from ImageNet-style natural scenes, providing a stringent test of out-of-distribution robustness.

NSD-Synthetic serves as a test-only evaluation. Because no training data exist for these synthetic stimuli, layer selection cannot be performed independently; instead, we inherit the best layer identified during the corresponding standard NSD evaluation for each (subject, region) pair. This design ensures that the OOD evaluation is not biased by layer selection on the synthetic stimuli themselves. fMRI responses were preprocessed identically to NSD (z-scored betas, averaged across repetitions).

### 4.3 TVSD (macaque electrophysiology)

The Temporal Visual Stream Dataset (Papale et al., 2025) provides multi-unit activity (MUA) recordings from 2 macaque monkeys across three cortical areas that form the core of the primate object recognition pathway: V1 (primary visual cortex), V4 (intermediate visual area), and IT (inferotemporal cortex). Including macaque data alongside human fMRI allows us to assess whether granularity effects on brain-model alignment generalise across species and recording modalities.

Stimuli are drawn from the THINGS object image set (Hebart et al., 2023). The training set comprises approximately 22,248 images (each presented once), while the test set contains 100 images presented 30 times each, with responses averaged across repetitions to obtain reliable estimates of neural tuning.

### 4.4 THINGS (behavioural similarity)

The THINGS dataset (Hebart et al., 2023) provides a complementary, behaviour-level measure of human visual representation. It comprises 1,854 diverse object concepts, each associated with a 66-dimensional embedding derived from large-scale human odd-one-out similarity judgments. These dimensions capture perceptual axes such as size, animacy, and texture, providing a structured measure of how humans perceive similarity across a broad range of everyday objects.

Unlike the neural datasets, THINGS operates at the concept level rather than the stimulus level. Each concept has multiple associated images; we pass all images through the model individually, then average the resulting activations within each concept to obtain a single concept-level representation before computing representational similarity. This concept averaging is essential because the behavioural embeddings are defined at the concept level, not the image level.

We use a fixed 80/20 concept-level train/test split (seed = 42), allocating approximately 370 concepts (20%) for layer selection and approximately 1,484 concepts (80%) for evaluation. Splitting at the concept level — rather than the image level — ensures that no concept appears in both the selection and evaluation sets, preventing information leakage between the two phases. THINGS behavioural data have no subject or region dimensions; a single alignment score is computed per model.

### Image preprocessing (all datasets)

All evaluation images — regardless of dataset — undergo the same preprocessing as training images (without augmentation): resize to 256 pixels along the shorter edge, centre-crop to $224 \times 224$, and normalisation with ImageNet channel statistics. This ensures that any differences in alignment are attributable to the model's learned representations, not to differences in input preprocessing.

---

## 5. Representational similarity analysis (RSA)

Representational similarity analysis (Kriegeskorte, 2008) provides an unweighted, geometry-preserving comparison between two representational spaces. Unlike encoding models that fit linear mappings (Section 6), RSA compares the *intrinsic structure* of representations without any learned transformation — making it sensitive to whether the model's representational geometry already resembles that of the brain, rather than whether brain-like information can be linearly extracted. This distinction is central to our study: if coarse-grained models achieve high RSA scores, it suggests that their representations are inherently more brain-like in structure, not merely that they contain recoverable information.

### 5.1 Representational dissimilarity matrices

For a given network layer (or brain region) and a set of $N$ stimuli, we construct an $N \times N$ representational dissimilarity matrix (RDM) in which each entry quantifies the dissimilarity between the representations of two stimuli:

$$\text{RDM}_{ij} = 1 - \rho(a_i, a_j)$$

where $\rho(a_i, a_j)$ is the Pearson correlation between the vectorized activation patterns for stimuli $i$ and $j$. Values range from 0 (identical representations) to 2 (maximally dissimilar). The Pearson correlation is computed via row-wise mean-centring and normalisation by the outer product of row standard deviations, with a numerical stabiliser of $10^{-12}$ added to prevent division by zero for constant-response stimuli.

### 5.2 RDM comparison

To quantify the alignment between a model RDM and a neural (or behavioural) RDM, we extract the upper triangle of each matrix (excluding the diagonal), yielding $N(N{-}1)/2$ pairwise dissimilarity values, and compute Spearman's rank correlation between the two vectors. Using rank correlation makes the comparison invariant to monotonic nonlinearities, capturing the ordinal structure of representational geometry rather than exact distance magnitudes. All reported RSA results use Spearman's $\rho$ unless otherwise noted.

### 5.3 Layer selection

Because different layers of a deep network capture features at different levels of abstraction, identifying the most neurally aligned layer is a necessary step before evaluating test-set performance. We perform layer selection on training data only, strictly separated from the test set, to prevent overfitting to the evaluation stimuli.

**NSD and TVSD.** For each (subject, region) pair, we randomly subsample 1,000 training stimuli (without replacement, seed = 42) — a subset large enough to yield stable RDM estimates while keeping computation tractable over the large NSD training sets (~9,000 stimuli per subject). For each of the 14 candidate extraction points, we construct a model RDM from SRP-projected activations and a neural RDM from the corresponding voxel (or channel) responses, then compare them via Spearman correlation. The extraction point yielding the highest correlation is selected as the best layer for that (subject, region) pair. Layer selection is performed independently for each subject and region, reflecting the possibility that different cortical areas may be best captured by different stages of the processing hierarchy.

**THINGS.** All ~370 concepts in the 20% selection split are used (no further subsampling). We compare concept-averaged SRP activations to the THINGS behavioural embedding via RSA for each candidate layer.

**NSD-Synthetic.** Because NSD-Synthetic is a test-only dataset with no associated training data, layer selection cannot be performed independently. Instead, we look up the best layer identified during the corresponding standard NSD evaluation for each (subject, region) pair and apply it directly to the synthetic stimuli. This ensures that the OOD evaluation reflects the model's generalisation capacity rather than an artefact of layer selection on unrepresentative stimuli.

### 5.4 Test evaluation with exact activations

After layer selection, the final RSA score is computed on held-out test data using exact, full-resolution activations — not SRP projections. For each unique best layer identified during selection, we run a fresh forward pass over the test stimuli, extracting only that single layer's activations without any dimensionality reduction. This re-extraction step ensures that the test-phase RDMs faithfully reflect the model's native representational geometry, unperturbed by the random projection used during layer selection.

The procedure is as follows:

1. Re-extract the selected layer's activations at full resolution from the test stimuli.
2. Construct the model RDM from these exact activations using Pearson correlation.
3. Construct the neural (or behavioural) RDM from the corresponding test-set responses.
4. Compute Spearman's $\rho$ between the upper triangles of the two RDMs.

Test set sizes are approximately 1,000 stimuli for NSD, 220 for NSD-Synthetic, 100 for TVSD, and 1,484 concepts for THINGS.

### 5.5 Bootstrap confidence intervals

We estimate uncertainty in RSA scores using a stimulus-level bootstrap procedure. On each of 1,000 iterations, we randomly draw 90% of the test stimuli without replacement (seed = 42), extract the corresponding submatrices from the pre-computed model and neural RDMs, and compute Spearman's $\rho$ on the submatrices. The 95% confidence interval is defined by the 2.5th and 97.5th percentiles of the resulting distribution. The full array of 1,000 bootstrap scores is stored for each evaluation, enabling subsequent statistical tests (e.g., pairwise comparisons between granularity conditions).

Subsampling without replacement (rather than with replacement, as in the classical bootstrap) is appropriate here because the test items are not exchangeable — each stimulus occupies a unique position in both the model and neural RDMs, and resampling with replacement would produce degenerate RDM entries on the diagonal.

---

## 6. Encoding score (voxelwise linear prediction)

While RSA tests whether the model's representational geometry already resembles that of the brain, encoding score analysis asks a complementary question: does the model's representation contain sufficient information to predict neural responses after a learned linear transformation? This distinction is conceptually important. A model with low RSA but high encoding score would indicate that brain-relevant information is present but not naturally emphasised in the model's geometry — it requires reweighting to align with the neural code. Conversely, a model with high RSA reflects an inherently brain-like representational structure. Running both analyses allows us to disentangle these two aspects of alignment and to test, in particular, whether coarse-grained training improves the intrinsic geometry (RSA) or the recoverable information content (encoding), or both.

### 6.1 Ridge regression

For each voxel (or recording channel), we fit a ridge regression from model activations to neural responses using `himalaya.ridge.RidgeCV`, a GPU-accelerated implementation that performs voxelwise mass-univariate regression with cross-validated regularisation parameter selection. The regularisation strength $\alpha$ is selected from 20 log-spaced candidates spanning $10^{-10}$ to $10^{10}$, using 5-fold cross-validation within the training set. The intercept is not fitted because all features are pre-normalised to zero mean (see below).

### 6.2 Feature and response normalisation

Both model activations and neural responses are z-normalised per feature (or voxel) dimension prior to fitting. To prevent data leakage, normalisation statistics (mean and standard deviation, with a stabiliser of $10^{-8}$) are computed exclusively from the training (or fit) split. Validation and test data are normalised using these same training-derived statistics. This protocol is applied consistently at both the layer selection and final evaluation stages.

### 6.3 Layer selection

Layer selection for encoding follows the same principle as for RSA — identify the most neurally predictive layer using training data only — but with a procedure tailored to the regression setting. For each (subject, region) pair, the training set is split 80/20 into fit and validation subsets using a seeded random permutation. For each of the 14 candidate extraction points:

1. SRP-projected activations are z-normalised using fit-split statistics.
2. Ridge regression with 5-fold CV is fitted on the fit split.
3. Predictions are generated for the validation split.
4. The selection score is the mean Pearson $r$ across all voxels between predicted and observed validation responses.

The layer achieving the highest mean validation Pearson $r$ is selected. Unlike RSA (which subsamples 1,000 training stimuli for efficiency), encoding layer selection uses the full 80% fit split, because the ridge regression fitting cost scales linearly with the number of features (after SRP projection to $k = 4096$) rather than quadratically with the number of stimuli (as in RDM construction).

### 6.4 Test evaluation

After identifying the best layer, the encoding model is refit on the **complete training set** (restoring the 20% validation data to maximise the training signal) for that single layer. Predictions are generated for the held-out test stimuli, and the encoding score is defined as the mean Pearson $r$ across all voxels between predicted and observed test responses.

Unlike RSA, encoding score uses SRP-projected activations ($k = 4096$) at all stages — layer selection and final evaluation alike. Re-extraction at full resolution is unnecessary here because the ridge regression is fitted on the projected features: the model learns a linear mapping in the SRP-projected space, and the Johnson–Lindenstrauss guarantee ensures that the quality of this linear mapping is approximately preserved under the random projection.

### 6.5 Bootstrap confidence intervals

Bootstrap CIs for encoding scores follow the same protocol as RSA: 1,000 iterations, 90% stimulus subsampling without replacement, and 95% CIs from the 2.5th and 97.5th percentiles. The key difference is efficiency: the ridge model is fitted once on the full training set, and the resulting test predictions are cached. Each bootstrap iteration merely subsamples the cached predictions and observed test responses, then recomputes the mean voxelwise Pearson $r$, without refitting the regression. This is valid because the bootstrap estimates uncertainty in the test-set correlation statistic, not in the model parameters.

### 6.6 Applicability

Encoding score evaluation is conducted for NSD and TVSD, which provide high-dimensional voxelwise (or channelwise) neural responses suitable for regression. It is not applicable to the THINGS behavioural dataset, which provides 66-dimensional concept-level embeddings rather than voxelwise data amenable to mass-univariate prediction.

---

## 7. PCA reconstruction control

A natural concern is that models trained on coarser labels might achieve higher alignment scores simply because they produce lower-dimensional representations — effectively filtering out high-frequency noise in the feature space — rather than because they learn qualitatively different visual features. To address this, we compare coarse-trained models against a PCA reconstruction baseline derived from the standard 1000-class model.

For each layer of the 1000-class model, we extract activations for the evaluation stimuli, project them onto the top $M$ principal components ($M = 1, 2, \ldots, 15$), and reconstruct the activations in this reduced subspace. We then compute RSA scores from these reconstructed activations and compare them to the scores achieved by models trained directly at the corresponding granularity level. If a $K$-way trained model outperforms the best $M$-component reconstruction of the 1000-class model, this constitutes evidence that the coarse-grained training objective induces representational structure that is not a simple low-rank projection of the fine-grained model's features — that is, the coarse-trained model has learned qualitatively distinct representations.

This control is conducted across NSD, TVSD, and THINGS, using the standard 1000-class model (3 seeds) with $M$ ranging from 1 to 15 components per layer.

---

## 8. Statistical reporting

### Confidence intervals

All reported confidence intervals are 95% bootstrap CIs derived from 1,000 stimulus-level iterations with 90% subsampling without replacement (Sections 5.5 and 6.5). For summary plots showing results averaged across subjects, error bars or shaded regions reflect the distribution across subjects and/or seeds (the exact statistic is specified per figure).

### Aggregation across seeds and subjects

Each experimental condition is evaluated with 3 independently trained models (seeds 1–3). For NSD (8 subjects) and TVSD (2 monkeys), alignment is computed independently per subject, with layer selection also performed per subject. Main figures report subject-averaged results; per-subject breakdowns are provided in supplementary materials.

### Complementary alignment metrics

RSA scores (Spearman's $\rho$ between model and neural RDMs) and encoding scores (mean voxelwise Pearson $r$) are not directly comparable in magnitude. RSA measures the geometric similarity of representational structure without any fitted transformation; encoding score measures variance explained after linear fitting. A model can in principle achieve high encoding score but low RSA if it contains brain-relevant information in a format that requires linear reweighting to match the neural code. We report both metrics to provide a complete picture of how granularity affects different aspects of brain-model alignment.

---

## 9. Computational resources

All model training and evaluation were conducted on a single NVIDIA RTX 4090 GPU (24 GB VRAM) with 32 CPU cores and 125 GB RAM. Training each model on the full ImageNet training set for 20 epochs required approximately [X] GPU-hours. Encoding score evaluation required approximately 8 GB of free VRAM per run due to the GPU-accelerated ridge regression (himalaya library with CUDA backend). The complete experimental pipeline — training 87 models from scratch and evaluating each across 4 datasets, multiple regions, 8 subjects (NSD), and 2 alignment analyses — represents a substantial computational investment designed to ensure comprehensive coverage of the experimental space.
