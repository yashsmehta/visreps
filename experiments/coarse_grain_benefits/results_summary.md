# Coarse-Grain Benefits: Results Summary

All experiments compare three AlexNet variants trained on ImageNet:
- **1K classes**: Standard 1000-way ImageNet training
- **64 classes**: Coarse-grained (64-way PCA labels from AlexNet)
- **64→1K curriculum**: Pre-trained on 64-way, then fine-tuned on 1000-way (late_layers mode, 10 epochs)

---

## 1. Few-Shot Learning (CIFAR-100 transfer)

Frozen fc2 features → logistic regression on CIFAR-100 with k shots per class (3 trials each).

| Model | k=1 | k=5 | k=10 | k=20 |
|-------|-----|-----|------|------|
| 1K classes | **10.9%** | **25.5%** | **32.4%** | **38.6%** |
| 64→1K curriculum | 10.2% | 22.6% | 27.9% | 33.8% |
| 64 classes | 8.1% | 17.2% | 22.6% | 28.7% |

**Takeaway:** 1K-way features transfer best. Coarse features lag ~10pp at k=20. Curriculum recovers ~50% of the gap.

---

## 2. ImageNet-C Robustness

Frozen fc2 features → linear probe on clean images → evaluate on 15 corruptions (severity=3, N=5000). Glass blur and fog excluded (no effect on any model).

| Model | Clean Acc | Mean Relative Robustness |
|-------|-----------|--------------------------|
| 1K classes | 19.4% | **42.9%** |
| 64→1K curriculum | 18.8% | 37.4% |
| 64 classes | 8.0% | 36.6% |

**Takeaway:** 1K-way retains the most accuracy under corruption. The 64-class model has much lower absolute accuracy (trained on only 64 classes), and slightly worse relative robustness.

---

## 2b. ImageNet-A Robustness (Natural Adversarial Examples)

Frozen fc2 features → linear probe on clean images (N=5000) → evaluate on ImageNet-A. Only 143 of 200 ImageNet-A classes overlap with our ImageNet variant; evaluation restricted to these.

| Model | Clean Acc (1K) | Clean Acc (143-class) | ImageNet-A Acc | Rel. Robustness |
|-------|---------------|----------------------|----------------|-----------------|
| 1K classes | 18.9% | 17.7% | 0.64% | 0.036 |
| 64→1K curriculum | 18.9% | 21.1% | **0.72%** | 0.034 |
| 64 classes | 7.9% | 8.7% | 0.45% | **0.051** |

**Takeaway:** All models perform near chance on ImageNet-A (~0.5-0.7%), consistent with the dataset being designed to fool standard architectures. Relative robustness is similar across models (~3-5%), suggesting ImageNet-A difficulty is dominated by the adversarial image selection itself, not label granularity. Less discriminative than ImageNet-C for this comparison.

---

## 3. Augmentation Invariance

Cosine similarity between clean and OOD-augmented fc2 features (1000 images, 10 augmentations each, albumentations).

| Model | Mean Cosine Similarity | Std |
|-------|----------------------|-----|
| 1K classes | **0.889** | 0.028 |
| 64→1K curriculum | 0.863 | 0.034 |
| 64 classes | 0.779 | 0.088 |

**Takeaway:** 1K-way representations are most stable. The 64-class model shows notably lower invariance with higher variance.

---

## 4. Curriculum Fine-Tuning (64→1K, late_layers)

Fine-tune conv5 + fc layers of the 64-way model on 1000-way ImageNet for 10 epochs.

| Epoch | Top-1 | Top-5 |
|-------|-------|-------|
| 0 | 0.1% | 0.5% |
| 2 | 48.5% | 73.4% |
| 4 | 55.5% | 78.8% |
| 6 | 60.6% | 82.5% |
| 8 | 64.4% | 84.8% |
| 10 | **65.7%** | **85.6%** |

**Takeaway:** Coarse pre-training provides a viable initialization — reaches 65.7% top-1 with only late layers unfrozen.

---

## 5. Curriculum NSD RSA (brain alignment)

RSA (Spearman) on NSD, all 7 layers, 8 subjects, early and ventral visual streams. Best layer scores averaged across subjects.

| Model | Early Visual (best layer) | Ventral Visual (best layer) |
|-------|---------------------------|----------------------------|
| 64 classes | **0.243** (conv4) | 0.217 (fc1) |
| 64→1K curriculum | **0.243** (conv4) | **0.246** (fc1) |
| 1K classes | 0.180 (conv3) | 0.238 (fc1) |

**Takeaway:** Curriculum model achieves best ventral stream alignment (0.246), surpassing both pure 64-class and pure 1K-class models. Early visual stream alignment is inherited from the frozen conv1-4 layers of the 64-class model.

---

## 6. Class Selectivity Index (CSI)

Per-neuron CSI = (mu_max - mu_other) / (mu_max + mu_other), computed on 1000-way ImageNet classes. Compares direct 1K-way vs. curriculum (64→1K).

| Model | conv1 | conv2 | conv3 | conv4 | conv5 | fc1 | fc2 |
|-------|-------|-------|-------|-------|-------|-----|-----|
| Direct 1K-way | 0.428 | 0.363 | 0.438 | 0.474 | 0.676 | 0.759 | **0.785** |
| Curriculum 64→1K | 0.440 | 0.387 | 0.462 | 0.506 | 0.676 | 0.749 | **0.684** |

**Takeaway:** Key difference is in fc2: curriculum model has lower CSI (0.684 vs 0.785) with much higher variance (std=0.37 vs 0.11), indicating more distributed, less class-selective representations in the final layer.

---

## 7. Linear Probe

Script exists (`linear_probe.py`) but no result CSV saved. Not yet run.
