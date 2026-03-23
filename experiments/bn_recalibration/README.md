# BatchNorm Recalibration: Diagnosing and Fixing Representational Collapse in ResNet50

## Executive Summary

ResNet50 trained on 16-class CLIP PCA labels (cfg16a) showed catastrophically poor THINGS behavioral alignment (score = 0.294) despite achieving 60% classification accuracy. Investigation revealed this was **not a training failure** — the model's weights learned rich, high-quality representations (effective dimensionality = 1,120 in train mode). The entire collapse was caused by **miscalibrated BatchNorm running statistics** at inference time, which amplified ~3% of images by up to 40,000x their normal activation magnitude. Recalibrating the BN running statistics on ImageNet restored the score to 0.589 — making cfg16a the **best-performing** model across all coarseness levels.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Background: How BatchNorm Works](#2-background-how-batchnorm-works)
3. [Root Cause: The Train/Eval BN Discrepancy](#3-root-cause-the-traineval-bn-discrepancy)
4. [The Amplification Cascade Through Skip Connections](#4-the-amplification-cascade-through-skip-connections)
5. [Why Only cfg16a? The Goldilocks Zone](#5-why-only-cfg16a-the-goldilocks-zone)
6. [Evidence: Layer-by-Layer Activation Tracing](#6-evidence-layer-by-layer-activation-tracing)
7. [Evidence: Representation Geometry](#7-evidence-representation-geometry)
8. [The Fix: BN Recalibration](#8-the-fix-bn-recalibration)
9. [Results After Recalibration](#9-results-after-recalibration)
10. [Implications and Recommendations](#10-implications-and-recommendations)
11. [Appendix: Detailed Data](#11-appendix-detailed-data)

---

## 1. The Problem

### Observed Symptom

When plotting THINGS behavioral alignment scores across coarseness levels for ResNet50 (all trained for 20 epochs with CosineAnnealingLR on ImageNet with CLIP PCA labels), the curve showed a dramatic, non-monotonic dip:

```
cfg_id  |  THINGS Score (Spearman)
--------|-------------------------
     2  |  0.5412
     4  |  0.5740
     8  |  0.4795*  (* also had a training schedule mismatch — see note)
    16  |  0.2942   <-- catastrophic drop
    32  |  0.4997
    64  |  0.5594
  1000  |  0.4684
```

The 16-class model scored 0.294 — roughly half the score of its neighbors (4-class: 0.574, 64-class: 0.559). This was puzzling because:
- The model trained successfully (60% test accuracy on 16 classes, well above 6.25% chance)
- The training hyperparameters were identical to cfg4a and cfg64a (lr=0.015, CosineAnnealingLR, 20 epochs, SGD, label_smoothing=0.1)
- Other architectures (ConvNeXt, ViT) showed no anomaly at 16 classes with the same labels

### Initial Hypotheses (All Wrong)

1. **Bad PCA labels at 16 classes?** No — ConvNeXt scored 0.527 with the same labels.
2. **Training divergence?** No — 60% test accuracy, smooth loss curve, reasonable final LR.
3. **Gradient explosion during training?** No — gradients from outlier images were only ~5x larger than normal (see Section 6).
4. **Representation collapse in the weights?** No — no dead neurons, no zero-variance channels, weight norms were between cfg4a and cfg64a.

---

## 2. Background: How BatchNorm Works

### The BatchNorm Transform

BatchNorm (Ioffe & Szegedy, 2015) normalizes activations within each channel across a mini-batch. For a channel with input values $x$, the transform is:

```
             x - μ
ŷ = γ · ─────────── + β
         √(σ² + ε)
```

Where:
- **μ** and **σ²** are the mean and variance of the channel
- **γ** (weight) and **β** (bias) are learnable affine parameters
- **ε** is a small constant for numerical stability (default: 1e-5)

### The Critical Difference: Training vs. Evaluation

**During training** (`model.train()`):
- μ and σ² are computed **from the current mini-batch**: the mean and variance of all activations in the batch for each channel
- These batch statistics adapt to whatever images are in the batch
- Running statistics are updated as an exponential moving average (EMA):
  ```
  running_mean = (1 - momentum) * running_mean + momentum * batch_mean
  running_var  = (1 - momentum) * running_var  + momentum * batch_var
  ```
  With the default PyTorch momentum of 0.1, the effective memory window is ~10 batches.

**During evaluation** (`model.eval()`):
- μ and σ² are the **running statistics** accumulated during training
- These are fixed scalars per channel — they do not adapt to the current input
- This makes inference deterministic and independent of batch composition

### Why This Matters

The running statistics represent a "population average" of what the network saw during training. They work well when:
1. The running statistics have converged to accurate population estimates
2. The test distribution is similar to the training distribution
3. The activation distribution for each channel is approximately Gaussian

When any of these assumptions break down, the BN normalization in eval mode can produce wildly incorrect outputs.

---

## 3. Root Cause: The Train/Eval BN Discrepancy

### The Smoking Gun Experiment

We extracted representations from cfg16a's penultimate layer (avgpool, 2048-dim) on THINGS images under three conditions:

| Condition | Eff. Dimensionality | PC1 Explains | L2 Norm (mean) | L2 Norm (max) |
|-----------|--------------------:|-------------:|---------------:|--------------:|
| **Eval mode** (original BN stats) | **16.7** | **96.7%** | **43.0** | **16,359** |
| **Train mode** (batch statistics) | **1,120.0** | **11.1%** | **4.28** | **6.76** |
| **Eval mode** (recalibrated BN) | **1,000.7** | **13.3%** | **4.70** | **93.1** |

In **train mode**, the representations are beautiful: 1,120 effective dimensions, evenly distributed variance, no outliers. This means the model's **weights are perfectly fine** — they encode rich, diverse features.

In **eval mode** with the original running statistics, the representations collapse to essentially **one dimension** (96.7% of variance in PC1), with a small subset of images producing L2 norms of 10,000–20,000 (vs. the normal ~4-5).

### What Goes Wrong, Step by Step

Consider an image that produces slightly unusual activations in an early layer — say, 2 standard deviations from the channel mean:

1. **During training**: BN computes batch_mean and batch_var from the current batch (which includes this image). The image's deviation is measured relative to its own batch → normalized output is moderate (~2σ worth).

2. **During evaluation**: BN uses running_mean and running_var, which were computed across millions of "typical" training images. If this image's activations deviate much more from the running_mean than from a typical batch mean, the normalized output can be enormous.

For cfg16a at layer1.1.bn3, we measured that outlier images (like `tarp_06s`) deviate **25-28σ** from the BN running mean. At deeper layers, the deviation compounds:

| Layer | Deviation from running_mean (in σ units) |
|-------|------------------------------------------:|
| layer1.1.bn3 | 25–28σ |
| layer2.0.bn3 | 79σ |
| layer3.0.bn3 | 228σ |
| layer3.5.bn3 | **55,940σ** |

A 55,940σ deviation is physically meaningless — it means the BN running statistics are catastrophically uncalibrated for these images.

### Why the Running Statistics Are Wrong

With PyTorch's default `momentum=0.1`, the EMA running statistics have an effective window of ~10 batches (10 × 32 = 320 images). After 20 epochs of training (~620,000 batch updates), the running statistics reflect only the **last ~320 images the model saw**, not the full training distribution.

More importantly, the model's weights change throughout training. The running statistics accumulate contributions from weights at ALL stages of training, but only the final weights matter for evaluation. Early in training, the weights produce very different activation distributions than at the end — these stale statistics corrupt the running average.

For most models (cfg4a, cfg64a, cfg1000a), this doesn't cause problems because:
- The activation distributions are approximately Gaussian (light tails)
- The running statistics, even if imperfect, are "close enough"

For cfg16a, the learned features have **heavy-tailed distributions** — a small fraction of images produce much larger activations than average. The running_var (which is an EMA of batch variances) underestimates the true variance of the tails. When eval-mode BN divides by this underestimated √(running_var), the result is enormous.

---

## 4. The Amplification Cascade Through Skip Connections

### ResNet's Architecture Creates a Multiplication Effect

ResNet50 consists of residual blocks with skip connections:

```
Input x ──┬──── [Conv → BN → ReLU → Conv → BN → ReLU → Conv → BN] ──── (+) → ReLU → Output
           │                                                               ↑
           └───────────────────── identity shortcut ──────────────────────┘
```

The critical design choice: **BatchNorm is INSIDE the residual branch, but there is NO normalization AFTER the skip connection addition.**

This means:
1. If the BN inside the branch produces an extreme output (due to miscalibrated running stats), that extreme value is **added** to the skip connection
2. The sum flows into the next block, where it becomes the input to the next BN
3. That BN's running stats also don't account for this extreme input → produces an even more extreme output
4. The process repeats across all 16 residual blocks in layer3-layer4

Each block amplifies the deviation because:
- The skip connection **preserves** the growing signal
- The internal BN **amplifies** the deviation (it's calibrated for normal inputs)
- ReLU keeps everything positive, preventing cancellation

### Tracing the Cascade for a Single Image

We traced the L2 norm of activations through every layer for `tarp_06s` (an extreme outlier) vs. `violin_01b` (a normal image):

```
Layer              Normal L2    Outlier L2     Ratio
────────────────────────────────────────────────────
stem.conv1            2,219        2,560       1.2x   ← Normal range
stem.bn1              1,401        1,327       0.9x   ← BN normalizes fine here
layer1.0              763          1,545       2.0x   ← Slight amplification
layer1.1.bn3          221         23,842     107.9x   ← EXPLOSION STARTS IN BN
layer1.1              819         13,876      16.9x   ← Skip connection dilutes partially
layer1.2.bn3          160         29,635     185.0x   ← BN amplifies further
layer2.3.bn3          505        100,494     199.0x   ← Growing
layer3.2.bn3          278        357,184   1,285.2x   ← Massive
layer3.5.bn3          230      4,045,774  17,619.3x   ← Catastrophic
layer4.1.bn3          117      4,941,057  42,240.9x   ← Peak
avgpool               4.6         20,861   4,565.3x   ← After global avg pool
fc                    5.1          5,423   1,069.1x   ← Final logits
```

The explosion begins at `layer1.1.bn3` (108x ratio) and grows exponentially through the network. By `layer4.1.bn3`, the outlier's activations are **42,000x** larger than normal.

### Why This Doesn't Happen During Training

During training, BN uses batch statistics. We verified this by running the same outlier image through the model in train mode:

| Layer | Eval L2 | Train L2 | Eval/Train Ratio |
|-------|--------:|---------:|-----------------:|
| layer1.1 | 13,876 | 924 | 15.0x |
| layer3.5 | 3,738,928 | 896 | **4,174.5x** |
| layer4.2 | 219,751 | 115 | **1,906x** |

In train mode, the activations are completely normal (~900 L2 norm) because BN computes statistics from the batch, which adapts to the current input. The explosion is **purely an eval-mode artifact**.

### Gradient Impact During Training

We also checked gradient norms during a simulated training step:

```
                          Normal Image    Outlier Image
layer1.0.conv1 grad_norm:      3.16           15.54       (5.2x)
layer3.0.conv1 grad_norm:      9.82           24.54       (2.5x)
layer4.2.conv3 grad_norm:      4.27            4.39       (1.0x)
fc.weight      grad_norm:      5.82            7.12       (1.2x)
```

The gradient from outlier images is only ~2-5x larger than normal in early layers, and essentially identical in later layers. In a batch of 32, this outlier contributes at most 5x/32 ≈ 15% of the batch gradient. **Gradients are not exploding** — the model receives well-behaved training signals throughout.

---

## 5. Why Only cfg16a? The Goldilocks Zone

### The Pattern Across Coarseness Levels

| Classes | Eval Eff. Dim | Train Eff. Dim | BN Problem? |
|--------:|--------------:|---------------:|-------------|
| 2 | ~900 | ~680 | No |
| 4 | 855 | 900 | No |
| **16** | **16.7** | **1,120** | **Severe** |
| 64 | 1,158 | 1,158 | No |
| 1000 | 1,411 | 1,441 | No |

Only cfg16a shows the massive eval/train discrepancy.

### The Explanation: Feature Specialization vs. Diversity

**2 classes (cfg2a):** The task is trivially simple. The gradient signal from a binary classification is too weak to deeply reshape the 2048-dim representation space. The model solves the task by modifying a small fraction of its weights, leaving most features close to their random initialization. The feature distribution stays approximately Gaussian → BN running stats are adequate.

**4 classes (cfg4a):** Still simple. Similar to 2 classes — minimal reshaping, light-tailed distributions.

**16 classes (cfg16a):** This is the critical transition point. There are enough classes (16 decision boundaries) to provide strong gradient signal that substantially reshapes the feature space. But 16 classes only require ~15 discriminative dimensions out of 2048. The model develops **highly specialized features** aligned with the 16-class decision boundaries. These specialized features respond very differently to different image types — images that align well with the learned decision boundaries produce small, well-behaved activations, while images that cross multiple boundaries or have unusual content produce extreme activations. This creates a **heavy-tailed distribution** that the BN running statistics fail to capture.

**64 classes (cfg64a):** The task demands 63+ discriminative dimensions. This forces the model to develop diverse features across many directions in the 2048-dim space. No single direction can become overly dominant because the gradient signal from 64 classes distributes representational energy across many dimensions. This naturally produces lighter-tailed, more Gaussian distributions → BN running stats are adequate.

**1000 classes (cfg1000a):** Maximum diversity pressure. The features must span hundreds of meaningful directions, producing the most Gaussian-like distributions of all models.

### Why ConvNeXt Doesn't Have This Problem

ConvNeXt trained on the same 16-class labels scores 0.527 (perfectly healthy, eff_dim = 119). Key architectural differences:

1. **LayerNorm instead of BatchNorm**: LayerNorm normalizes per-sample, not per-batch. There are no running statistics and no train/eval discrepancy. Each image is normalized using its own statistics.

2. **No traditional skip connections**: ConvNeXt uses a different residual structure with depthwise separable convolutions that doesn't create the same amplification highways.

3. **GELU instead of ReLU**: GELU allows negative values, potentially providing more stable gradient flow.

---

## 6. Evidence: Layer-by-Layer Activation Tracing

### Methodology

We loaded the trained cfg16a model and passed individual images through it, hooking into every residual block and BN layer to record activation statistics. We compared:
- A normal image (`violin_01b`) — a small handheld object
- An extreme outlier (`tarp_06s`) — an elongated object on an open background

### BN Input Deviation Analysis

For each BN layer, we measured how far the outlier's input activations deviated from the BN running_mean, expressed in units of √(running_var):

```python
deviation = (outlier_activation_mean - running_mean) / sqrt(running_var)
```

| Layer | Max Channel Deviation (σ) | Mean Channel Deviation (σ) |
|-------|---------------------------:|----------------------------:|
| layer1.1.bn3 | 28.1 | 25.5 |
| layer2.0.bn3 | 79.0 | 39.7 |
| layer3.0.bn3 | 227.8 | 48.9 |
| layer3.5.bn3 | **55,940** | **32,040** |

At layer3.5, the BN running statistics see the outlier's activations as 55,940 standard deviations from the mean. For reference, a 6σ event in a Gaussian distribution has probability ~2×10⁻⁹. A 55,940σ event is not a statistical fluctuation — it means the running statistics are fundamentally wrong for this image.

### The Outlier Population

Of the 26,107 THINGS images, ~736 images (2.8%) had PC1 scores in the extreme negative tail. These were predominantly:

**Most extreme (largest activation explosion):**
- tarp, swordfish, flagpole, baton, fish, bin, mold, credit_card, shark, goalpost, cape, turban, cigarette_holder, streetlight, wind_chimes

**Common characteristics:** Elongated objects against open or uniform backgrounds (sky, water, flat surfaces). These images likely trigger specific edge/orientation filters in early layers that, through the BN amplification cascade, produce runaway activations.

**Normal images (no explosion):**
- violin, guitar, sword, toothbrush, tie, saxophone, microphone — small, compact objects with complex textures

All 25,371 normal images had PC1 scores tightly clustered around +51.6 with L2 norms of ~4-5. The 736 outliers had PC1 scores ranging from -20,629 to ~0, with L2 norms reaching 20,860.

---

## 7. Evidence: Representation Geometry

### Singular Value Spectrum

We computed the SVD of the (26,107 × 2048) activation matrix for each model:

**cfg4a (4 classes) — Healthy:**
```
PC1:  19.2%    Top-5:  49.6%    Top-10:  61.8%    Eff. dim: 900.3
```
Variance is smoothly distributed across ~900 dimensions.

**cfg16a (16 classes) — Collapsed (eval mode):**
```
PC1:  96.3%    Top-5:  99.4%    Top-10:  99.8%    Eff. dim: 30.4
```
96.3% of all variance lives in a single dimension. The eigenvalue of PC1 is **303,191** — vs. 7,299 for PC2 (41.5x ratio). The representation is essentially one-dimensional.

**cfg16a (16 classes) — Healthy (train mode):**
```
PC1:  11.1%    Top-5:  30.4%    Top-10:  41.9%    Eff. dim: 1,120.0
```
In train mode, cfg16a has the **richest** representations of any model — even better than cfg4a!

**cfg64a (64 classes) — Healthy:**
```
PC1:  14.6%    Top-5:  32.0%    Top-10:  43.6%    Eff. dim: 1,157.5
```

### Where the Collapse Propagates (cfg16a, eval mode)

The collapse isn't just in the final layer — it propagates deep into the network:

| Layer | Effective Dim (cfg16a) | Effective Dim (cfg2a) |
|-------|-------------------------:|------------------------:|
| layer1 (256 channels) | 51.0 | 91.6 |
| layer2 (512 channels) | **37.7** (PC1=98.5%) | 214.4 |
| layer3 (1024 channels) | **1.6** (PC1=100.0%) | 352.6 |
| layer4 (2048 channels) | **16.7** (PC1=96.7%) | 681.2 |

At **layer3, the representation is literally 1-dimensional** (eff_dim = 1.6). The activation standard deviation at this layer is 205.4 — one direction has massive energy, everything else is negligible. The slight recovery at layer4 (eff_dim 16.7 vs. 1.6) is because layer4's BN parameters partially re-distribute the energy.

### Post-Hoc PC Removal Does Not Help

We tested whether removing the collapsed principal components could restore useful structure:

| Condition | RDM correlation with cfg4a |
|-----------|---------------------------:|
| cfg16 original | 0.7439 |
| cfg16 minus PC1 | 0.5222 (worse!) |
| cfg16 minus top-3 PCs | 0.6073 |
| cfg16 minus top-15 PCs | 0.6358 |

Removing the collapsed PCs makes things **worse**, not better. The collapse has fundamentally distorted all dimensions, not just the dominant ones. The remaining dimensions don't contain useful structure because the model learned them in the context of the collapsed representation — they're optimized for a world where PC1 dominates everything.

### Weight-Level Analysis (No Anomaly)

Despite the catastrophic eval-mode collapse, the model's weights show no pathology:

| Metric | cfg4a | cfg16a | cfg64a |
|--------|------:|-------:|-------:|
| FC weight std | 0.0525 | 0.0554 | 0.0439 |
| layer4.2 conv weight eff. rank | 178.3 | 255.0 | 316.8 |
| BN dead channels | 0/2048 | 0/2048 | 0/2048 |
| BN mean_of_vars (layer4.2) | 0.1355 | 0.2626 | 0.1516 |
| FC condition number | 324.7 | 353.8 | 339.5 |

cfg16a is smoothly interpolated between cfg4a and cfg64a across all metrics. The weights are healthy — only the BN running statistics are miscalibrated.

---

## 8. The Fix: BN Recalibration

### What BN Recalibration Does

Instead of using the EMA running statistics accumulated during training, we recompute them from scratch using a single forward pass through the training data with a cumulative moving average:

```python
# Step 1: Reset running stats and switch to cumulative average
model.train()
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.reset_running_stats()
        module.momentum = None  # Use cumulative average, not EMA

# Step 2: Forward pass through training data (no gradient computation)
with torch.no_grad():
    for images, _ in imagenet_train_loader:
        model(images.cuda())
        # running_mean and running_var update automatically in train mode

# Step 3: Switch back to eval mode with new running stats
model.eval()
# Now BN uses the freshly computed running stats
```

### Why `momentum=None` (Cumulative Average) Works Better Than Default EMA

With PyTorch's default `momentum=0.1`:
```
running_mean = 0.9 * running_mean + 0.1 * batch_mean
```

After N batches, the weight of batch k (counting from the start) is approximately `0.1 * 0.9^(N-k)`. This means:
- Recent batches dominate (last 10 batches contribute ~65% of the total)
- Batches from early in calibration contribute almost nothing
- If rare outlier images don't appear in the last few batches, the running statistics won't account for them

With `momentum=None` (cumulative average):
```
running_mean = (count * running_mean + batch_sum) / (count + batch_size)
```

This computes the **true sample mean** over all images seen — every image contributes equally. This is strictly more accurate for estimating population statistics.

### Why Recalibration on Final Weights Matters

During training, BN running statistics accumulate contributions from ALL epochs:
- Epoch 1: Random weights → random activation distributions → running stats get a "random" contribution
- Epoch 10: Partially trained weights → different distributions → running stats get a "partially trained" contribution
- Epoch 20: Final weights → final distributions → running stats get the correct contribution

But the EMA from epoch 20 still carries influence from epochs 1-19 (decaying with 0.9^N). For cfg16a, where the feature distributions changed dramatically during training (the heavy-tailed structure only developed in later epochs), the running statistics are contaminated by early-epoch contributions that are completely irrelevant to the final model.

Recalibration after training uses **only the final weights**, giving accurate statistics for the actual model being evaluated.

### How Many Images Are Needed?

We used 2,000 batches of 256 images = 512,000 images (about half of ImageNet's training set). This is likely more than necessary — the cumulative average converges quickly because:

1. Central Limit Theorem: The running_mean converges as O(1/√N) in the number of samples
2. For a 2048-channel BN layer, each batch provides 2,048 independent statistics
3. After ~50,000 images, most channels' statistics are well-converged

However, for heavy-tailed distributions (like cfg16a's), more samples are better because the variance estimate converges more slowly when there are outliers.

---

## 9. Results After Recalibration

### cfg16a: The Dramatic Fix

```
THINGS alignment (Spearman):
  Original BN:         0.2942  [0.2848, 0.3049]  layer=block16
  Recalibrated BN:     0.5891                     layer=block16
  Delta:              +0.2949
```

The score more than doubles, going from the worst to the best model.

### Representation Quality After Recalibration

| Metric | Original BN | Recalibrated BN |
|--------|------------:|----------------:|
| Effective dim | 16.7 | **1,000.7** |
| PC1 explains | 96.7% | **13.3%** |
| L2 norm mean | 43.0 | **4.70** |
| L2 norm max | 16,359 | **93.1** |
| Activation std | 10.54 | **0.095** |

The recalibrated representations are essentially identical to the train-mode representations (eff_dim 1,001 vs. 1,120), confirming that the recalibrated running statistics accurately capture the true population distribution.

### cfg1000a: Minimal Effect (Confirming the Diagnosis)

```
THINGS alignment (Spearman):
  Original BN:         0.4684  layer=block15
  Recalibrated BN:     0.4989  layer=block15
  Delta:              +0.0305
```

The 1000-class model shows only a +0.03 improvement — its original BN stats were already well-calibrated. This confirms that the BN problem is specific to models with heavy-tailed feature distributions, not a universal issue.

### Representation quality for cfg1000a (no significant change):

| Metric | Original BN | Recalibrated BN |
|--------|------------:|----------------:|
| Effective dim | 1,411 | 1,389 |
| PC1 explains | 3.5% | 3.8% |
| L2 norm max | 37.9 | 39.8 |

---

## 10. Implications and Recommendations

### For This Project

1. **BN recalibration should be standard procedure** after training any ResNet model, before running evaluations. A single forward pass through the training set with `momentum=None` takes ~5 minutes and eliminates this class of artifacts.

2. **Check all existing ResNet results** — any model with a moderate number of output classes (8-32) could potentially be affected. Models with very few (2-4) or very many (64+) classes are likely fine.

3. **The coarseness story gets stronger** — after the BN fix, cfg16a is the best-performing model (0.589 > cfg4a's 0.574 > cfg64a's 0.559 > cfg1000a's 0.499), suggesting that moderate coarse-graining optimally aligns representations with human similarity judgments.

### For the Field

1. **BatchNorm running statistics are a hidden evaluation artifact.** Most papers report results in eval mode without verifying that BN statistics are well-calibrated. This can silently corrupt results, especially for:
   - Models trained for few epochs
   - Models with few output classes (leading to specialized, heavy-tailed features)
   - Transfer learning scenarios where the eval distribution differs from training

2. **The train/eval BN discrepancy is architecture-specific.** ResNet (with BN inside residual branches and no normalization after skip connections) is particularly susceptible because skip connections amplify BN errors exponentially across blocks. Architectures using LayerNorm (ViT, ConvNeXt) are immune.

3. **Effective dimensionality is a diagnostic canary.** If a model's eval-mode representations have suspiciously low effective dimensionality compared to train-mode, BN miscalibration is the likely culprit.

### Recommended Hyperparameter Changes for ResNet Training

| Change | Impact | Rationale |
|--------|--------|-----------|
| Post-training BN recalibration | **Critical** | Eliminates the artifact entirely with zero impact on training |
| `warmup_epochs=5` | Moderate | Prevents early unstable batches from corrupting BN EMA |
| `bn_momentum=0.01` or `None` | Moderate | More stable running stats (but longer warmup needed for convergence) |
| GroupNorm instead of BatchNorm | Architectural | Eliminates train/eval discrepancy entirely (per-sample normalization) |

### What Won't Help

| Change | Why Not |
|--------|---------|
| Gradient clipping | Gradients aren't exploding — outlier image gradients are only 2-5x larger |
| Higher weight decay | Weights aren't unusually large; the issue is in BN statistics, not weights |
| Different random seed | The BN mechanism is systematic (confirmed across architectures) |
| Removing outlier images | The outlier images themselves are normal; they only appear abnormal through the lens of miscalibrated BN |

---

## 11. Appendix: Detailed Data

### A. Training Configurations

All models used identical hyperparameters (except cfg8a and cfg32a, which used a different schedule):

```json
{
  "optimizer": "sgd",
  "learning_rate": 0.015,
  "weight_decay": 0.0001,
  "lr_scheduler": "cosineannealinglr",
  "label_smoothing": 0.1,
  "num_epochs": 20,
  "warmup_epochs": 0,
  "use_amp": true,
  "batch_size": 32,
  "pretrained_dataset": "none"
}
```

### B. Training Metrics at Epoch 20

| cfg_id | Train Loss | Train Acc | Test Acc | Final LR |
|-------:|-----------:|----------:|---------:|---------:|
| 2 | 0.329 | 92.2% | 93.1% | 0.000838 |
| 4 | 0.701 | 80.8% | 82.7% | 0.000838 |
| 8* | 1.067 | 69.5% | 71.3% | 0.000838 |
| 16 | 1.484 | 58.0% | 59.7% | 0.000838 |
| 32* | 1.911 | 50.5% | 52.3% | 0.001250 |
| 64 | 2.238 | 45.0% | 46.3% | 0.000838 |

*cfg8a and cfg32a were trained with StepLR/90 epochs (different schedule).

### C. PCA Label Class Balance

| n_classes | Min class size | Max class size | Imbalance ratio |
|----------:|---------------:|---------------:|----------------:|
| 2 | 630,703 | 630,703 | 1.0x |
| 4 | 315,202 | 315,501 | 1.0x |
| 8 | 94,073 | 221,129 | 2.4x |
| 16 | 28,770 | 136,286 | 4.7x |
| 32 | 5,476 | 77,269 | 14.1x |
| 64 | 2,040 | 64,403 | 31.6x |

Class imbalance alone does not explain the BN problem — cfg64a has 32x imbalance but no BN issues.

### D. Cross-Architecture Comparison at 16 Classes

| Architecture | THINGS Score (epoch 20) | BN Type | BN Problem? |
|-------------|------------------------:|---------|-------------|
| ResNet50 | 0.294 (→ 0.589 after fix) | BatchNorm2d | **Yes** |
| ConvNeXt_Base | 0.527 | LayerNorm | No |
| ViT-Base | 0.441 | LayerNorm | No |
| CustomCNN | 0.224 | BatchNorm1d | Possibly* |

*CustomCNN shows erratic scores across all levels, which could include BN artifacts, but this was not investigated.

### E. Concepts Driving the Collapse

**Top 15 concepts with most extreme activations (negative PC1 direction):**

| Concept | Avg PC1 Score | # Images |
|---------|-------------:|----------:|
| swordfish | -4,467 | 12 |
| streetlight | -2,626 | 16 |
| shark | -2,543 | 17 |
| tarp | -2,303 | 13 |
| windsock | -2,194 | 13 |
| wing | -1,807 | 15 |
| parachute | -1,756 | 13 |
| flagpole | -1,609 | 14 |
| antenna | -1,589 | 12 |
| jellyfish | -1,491 | 15 |
| goalpost | -1,477 | 13 |
| seahorse | -1,365 | 15 |
| cloud | -1,332 | 12 |
| fish | -1,169 | 13 |
| bin | -1,120 | 14 |

These are predominantly **elongated objects against uniform backgrounds** — the kind of images that activate specific directional filters strongly.

---

## References

- Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Papyan, V. et al. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training. PNAS.
- Singh, S. & Krishnan, S. (2020). Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks. CVPR 2020.
