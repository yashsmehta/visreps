# NSD synthetic: reliable-voxel OOD RSA

Completed 2026-09-06 on the local machine. CustomCNN CLIP-32 versus default-1000, epoch 20, training seed 1, all eight NSD subjects.

CLIP-32 scores higher in the early visual stream for all eight subjects; default-1000 scores higher in the ventral stream for all eight subjects. These rankings reverse the regular-NSD rankings from the preceding dataset-calibrated comparison.

| Region | CLIP-32 | Default-1000 | CLIP-32 minus default | Layers (CLIP / default) |
|---|---:|---:|---:|---|
| early visual stream | 0.2720 | 0.1428 | +0.1291 | fc1 / conv4 |
| ventral visual stream | 0.2294 | 0.2697 | -0.0403 | fc1 / fc2 |

Protocol follows the historical NSD-synthetic test-only setup recovered from commit `f907e69`: regular NSD training data for selection, all 220 shared synthetic-dataset stimuli for testing. The current single-layer-per-region rule is used, rather than the old per-subject selection rule. Synthetic images are never used for BatchNorm calibration or layer selection. The 220-image set is the historical full stimulus battery, including its natural-scene controls; no category was removed.

BatchNorm: default training-only calibration on 69,566 regular NSD images, excluding all participating subjects’ NSD test images. Reused the exact validated, dataset-specific BN caches from the preceding proof-of-principle experiment. Model parameters, original buffers, calibrated buffers, and calibration-image identities were checked against saved SHA-256 hashes. Models remained in eval mode for extraction.

Layer selection: reused the immediately preceding regular-NSD evaluation with the same checkpoints, BN buffers, NCSNR-filtered responses, and subject set. Each region uses the layer with highest mean training selection RSA across subjects. Selection scores are retained in the raw result JSON files. No selection was performed on synthetic scores.

Voxel reliability: subject-specific regular-NSD NCSNR strictly greater than 0.2 (finite values only), at 1.8-mm resolution, fithrf_GLMdenoise_RR. This is an independent natural-NSD reliability criterion, not synthetic-test-derived reliability. Intersected with each synthetic ROI’s brain/validity mask. The original synthetic archive omitted voxel coordinates; recovered its exact voxel ordering from the original loader’s brain mask, synthetic validity mask, and ROI definitions. Checked ROI voxel counts and coordinate ordering, then verified the first, middle, and last voxel against raw repeated-response betas in each subject/region (48 checks, all passed). Filtered responses are saved separately in `reliable_synthetic.pkl`; original data are unchanged.

| Subject index | Early retained / total | Ventral retained / total |
|---|---:|---:|
| 0 | 3909/5917 | 3531/7604 |
| 1 | 3058/4611 | 4132/8185 |
| 2 | 3284/5148 | 3528/8167 |
| 3 | 2769/4054 | 3427/7645 |
| 4 | 3636/4494 | 3371/6619 |
| 5 | 4857/5860 | 4437/9666 |
| 6 | 2804/4559 | 2554/6515 |
| 7 | 2415/4596 | 2654/8350 |

RSA: Pearson-distance RDMs, Spearman correlation between upper triangles, exact post-activation selected-layer features without projection/PCA reconstruction at test time. Neural responses are the existing repetition-averaged, z-scored synthetic betas. Reported means are arithmetic averages across eight subjects. No bootstrap or significance testing; one trained seed.

Database: `log_expdata=false`, writer disabled, and identical before/after SHA-256 recorded in `database_check.json`. No checkpoint writes.

Reproduce from repository root:

```bash
PYTHONPATH=. .venv/bin/python experiments/nsd_synthetic_poc/filter_voxels.py
OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python experiments/nsd_synthetic_poc/evaluate.py
```

The evaluator uses the local NSD selection results and metadata in `experiments/bn_recalibration/dataset_stats_comparison/`. Raw results, selected layers, effective configs, paired differences, voxel counts, and verification logs are alongside this report.

---

# Addendum: corrected stimulus transform (2026-09-06)

The section above used `visreps.dataloaders.obj_cls.get_transform()` to turn the synthetic
PNGs into model inputs. `audit_preprocessing.py` compared that against the dataset authors'
own feature-extraction recipe and found it wrong for this stimulus set.

The raw synthetic images are `714 x 1360` uint8: a centred `714 x 714` content square with
grey padding either side. The two pipelines differ in two ways.

| step | legacy `get_transform()` | authors' reference |
|---|---|---|
| gamma | none | `sqrt(uint8/255)*255` linearisation |
| crop | `Resize(256)` on the short side, then `CenterCrop(224)` | `CenterCrop(min(size)) = 714 x 714`, then `Resize(224)` |

`Resize(256)` scales the 714-pixel short side to 256, giving `256 x 488`; the subsequent
`CenterCrop(224)` therefore keeps only the central 87.5% of the content square and discards
a 45-pixel band on every side. That band is where the `word4_pos*` / `word6_pos*` stimuli put
their words: `word4_pos2_1` renders as a clipped "eet" under the legacy transform and a
complete "feet" under the reference. The whole point of those 40 stimuli is retinal position,
so the legacy transform destroys the manipulation it is meant to probe. See
`preprocessing_audit/input_comparison.png`.

16 of 220 stimuli (`word{4,6}_pos{1,5}_*`) are spatially constant after cropping under *both*
pipelines - those words fall entirely outside the content square. That is a property of the
authors' recipe, not a bug introduced here, and it is reproduced faithfully.

`audit_preprocessing.py` verifies the fix end to end: all 220 saved PNGs are bit-identical to
`nsdsynthetic_stimuli.hdf5`, and all 220 `SyntheticReferenceTransform` outputs match an
independent explicit PIL-crop / tensor-arithmetic oracle at `rtol=0, atol=0`. The expected
input tensors are frozen in `preprocessing_audit/expected_model_inputs.pt`, and
`evaluate.py --corrected-imagenet` re-checks every image against them through a forward
pre-hook during the real evaluation (440 checks = 220 images x 2 distinct selected layers).

## Corrected results

`evaluate.py --corrected-imagenet` -> `corrected_imagenet/`. It changes the stimulus transform
**and** the BatchNorm source together: it uses the reference transform and the checkpoints'
original ImageNet BN buffers, reusing the `saved_imagenet` layer selection from
`experiments/bn_recalibration/dataset_stats_comparison/`.

| Region | Condition | CLIP-32 | Default-1000 | CLIP-32 minus default | Subjects with CLIP-32 higher | Layers |
|---|---|---:|---:|---:|---:|---|
| early | legacy transform, NSD-recalibrated BN | 0.2720 | 0.1428 | +0.1291 | 8/8 | fc1 / conv4 |
| early | reference transform, ImageNet BN | 0.1607 | 0.1504 | +0.0103 | 8/8 | conv4 / conv4 |
| ventral | legacy transform, NSD-recalibrated BN | 0.2294 | 0.2697 | -0.0403 | 0/8 | fc1 / fc2 |
| ventral | reference transform, ImageNet BN | 0.2650 | 0.2575 | +0.0075 | 6/8 | fc1 / fc1 |

The headline claim of the section above does not survive. Under the corrected condition the
ventral reversal disappears: CLIP-32 is no longer beaten by default-1000 in ventral cortex, it
edges ahead by +0.0075 (6/8 subjects). The early-stream advantage survives in direction and
unanimity but shrinks by an order of magnitude, from +0.1291 to +0.0103. Both models now
select the same layer in both regions, where previously they selected different ones.

**This comparison confounds two changes.** The corrected run alters the transform and the BN
source simultaneously, so the collapse of the ventral reversal cannot be attributed to the
transform alone. Disentangling it needs a fourth cell - reference transform with
NSD-recalibrated BN - which `evaluate.py` cannot currently express, because `--corrected-imagenet`
couples the two switches. Until that cell exists, neither condition in the table should be
quoted as the NSD-synthetic result.

Both conditions remain single-seed, no bootstrap, no significance testing. `results.db` was
verified unchanged by SHA-256 before and after (`corrected_imagenet/database_check.json`).

```bash
PYTHONPATH=. .venv/bin/python experiments/nsd_synthetic_poc/audit_preprocessing.py
OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python experiments/nsd_synthetic_poc/evaluate.py --corrected-imagenet
```

---

# Addendum 2: the missing cell — reference transform + NSD-recalibrated BN (2026-09-06)

Run through the integrated pipeline (`neural_dataset=nsd_synthetic bn_calibration_source=nsd`),
which reuses each ROI's layer and the cached BN statistics from the matching regular-NSD run in
`results.db`. Seed 1, no bootstrap. Results in `reference_nsdbn/`.

| Region | Transform | BN | CLIP-32 | Default-1000 | Δ | CLIP higher | Layers |
|---|---|---|---:|---:|---:|---:|---|
| early | legacy | NSD | 0.2720 | 0.1428 | +0.1291 | 8/8 | fc1 / conv4 |
| early | reference | NSD | 0.3328 | 0.1337 | **+0.1991** | 8/8 | fc1 / conv4 |
| early | reference | ImageNet | 0.1607 | 0.1504 | +0.0103 | 8/8 | conv4 / conv4 |
| ventral | legacy | NSD | 0.2294 | 0.2697 | −0.0403 | 0/8 | fc1 / fc2 |
| ventral | reference | NSD | 0.2718 | 0.2764 | **−0.0046** | 3/8 | fc1 / fc2 |
| ventral | reference | ImageNet | 0.2650 | 0.2575 | +0.0075 | 6/8 | fc1 / fc1 |

With the third cell the two earlier changes separate cleanly:

- **The ventral reversal was the transform.** Holding BN fixed at NSD and fixing only the crop
  takes it from −0.040 (0/8 subjects) to −0.005 (3/8): a null. The legacy crop, which clipped
  the word-position stimuli, manufactured the "default-1000 wins in ventral" result.
- **The early-stream advantage is the BN source.** Holding the transform fixed at reference and
  switching BN from ImageNet to NSD takes it from +0.010 to +0.199. Under NSD statistics CLIP-32
  selects fc1 for early cortex and scores 0.33 on synthetic stimuli; under ImageNet statistics it
  selects conv4 and scores 0.16. Default-1000 is on conv4 either way and barely moves.

Each row is internally consistent (the layer was selected under the same BN it is evaluated
with), so the remaining question is which BN condition is the right one to report, not whether
the pipeline is correct. Still single-seed and unbootstrapped.

## Decomposing the early-stream gap: layer choice vs BN statistics

`reference_nsdbn/pinned_layer_x_bn.py` pins the layer and crosses the BN source (reference
transform throughout, seed 1). Mean RSA across 8 subjects:

| Region | Model | Layer | ImageNet BN | NSD BN | NSD − ImageNet |
|---|---|---|---:|---:|---:|
| early | CLIP-32 | conv4 | 0.1607 | 0.1530 | −0.008 |
| early | CLIP-32 | **fc1** | 0.2614 | **0.3328** | **+0.071** |
| early | default-1000 | conv4 | 0.1504 | 0.1337 | −0.017 |
| early | default-1000 | fc1 | 0.1796 | 0.1644 | −0.015 |
| ventral | CLIP-32 | conv4 | 0.2672 | 0.2579 | −0.009 |
| ventral | CLIP-32 | fc1 | 0.2650 | 0.2718 | +0.007 |
| ventral | default-1000 | conv4 | 0.2638 | 0.2481 | −0.016 |
| ventral | default-1000 | fc1 | 0.2575 | 0.2324 | −0.025 |

CLIP-32's early-stream score goes from 0.161 (ImageNet BN, conv4) to 0.333 (NSD BN, fc1).
That +0.172 splits into **+0.100 from the layer** (fc1 beats conv4 on synthetic stimuli even
under ImageNet BN, but ImageNet-BN selection on natural NSD picked conv4) and **+0.071 from
the BN statistics** acting on fc1 specifically. NSD BN is not a general boost: it lowers every
other cell in the table, including all four default-1000 cells, by 0.01–0.025. The only
representation it helps is CLIP-32's fc1 in early cortex.

Also visible: on synthetic stimuli fc1 outscores conv4 in early cortex for *both* models under
ImageNet BN, yet natural-NSD selection chose conv4 for both. The inherited layer is the
natural-image optimum, not the synthetic optimum; that is what an OOD test is supposed to do.

---

# Addendum 3: in-dataset layer selection on a 50/50 split (2026-09-06)

`layer_source=split`: the 16 blank word stimuli are dropped (204 remain), then a fixed
stratified split (seed 42, half of every stimulus family) gives 102 stimuli for layer
selection and 102 for reporting. Reference transform, seed 1, no bootstrap. Both BN sources.
Results in `split_selection/`.

## Per-layer selection scores (mean over 8 subjects, selection half)

| Region | BN | Model | conv1 | conv2 | conv3 | conv4 | conv5 | fc1 | fc2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| early | ImageNet | CLIP-32 | −0.108 | −0.046 | 0.000 | 0.082 | 0.270 | 0.192 | **0.272** |
| early | ImageNet | default-1000 | −0.108 | −0.068 | −0.033 | 0.092 | **0.254** | 0.112 | 0.156 |
| early | NSD | CLIP-32 | −0.096 | −0.041 | −0.004 | 0.072 | **0.276** | 0.246 | 0.196 |
| early | NSD | default-1000 | −0.100 | −0.054 | −0.036 | 0.068 | **0.227** | 0.096 | 0.133 |
| ventral | ImageNet | CLIP-32 | −0.074 | 0.030 | 0.119 | 0.257 | **0.295** | 0.250 | 0.287 |
| ventral | ImageNet | default-1000 | −0.083 | 0.005 | 0.078 | 0.256 | 0.265 | 0.247 | **0.289** |
| ventral | NSD | CLIP-32 | −0.066 | 0.048 | 0.119 | 0.248 | **0.260** | 0.258 | 0.236 |
| ventral | NSD | default-1000 | −0.078 | 0.017 | 0.080 | 0.242 | 0.258 | 0.218 | **0.262** |

## Reported scores (held-out half)

| Region | BN | CLIP-32 | Default-1000 | Δ | CLIP higher | Layers |
|---|---|---:|---:|---:|---:|---|
| early | ImageNet | 0.2732 | 0.1717 | **+0.1015** | 8/8 | fc2 / conv5 |
| early | NSD | 0.2452 | 0.1532 | **+0.0920** | 8/8 | conv5 / conv5 |
| ventral | ImageNet | 0.2793 | 0.2849 | −0.0055 | 2/8 | conv5 / fc2 |
| ventral | NSD | 0.2466 | 0.2584 | −0.0117 | 2/8 | conv5 / fc2 |

- **Early cortex: CLIP-32 wins by ~+0.10 in 8/8 subjects regardless of BN source**, and the
  NSD-BN row has both models on conv5, so this is a like-for-like comparison at matched depth.
  In-dataset selection shrinks the inherited-layer gap (+0.20 under NSD BN) but leaves a
  robust effect.
- **Ventral cortex: null**, agreeing with every reference-transform protocol so far.
- **BN source matters far less once selection sees synthetic stimuli**: the +0.20 vs +0.01
  swing under inherited layers becomes +0.10 vs +0.09 here.
- **Selection is near-tied for CLIP-32 early under ImageNet BN**: fc2 0.272 vs conv5 0.270.
  With 102 stimuli that is a coin flip; the reported score would be ~0.245 rather than 0.273
  had conv5 won. Treat the specific layer, not the effect, as unstable.
- **conv1–conv3 score at or below zero in early cortex for both models.** The usual early
  cortex ↔ early layer pairing does not hold on this stimulus set; nothing before conv5
  explains early-visual synthetic responses. Unexplained, and worth its own look.

## Untrained baseline (same architecture, epoch 0, seed 1)

`split_selection/evaluate_untrained.py`: `/data/ymehta3/default/cfg1000a/checkpoint_epoch_0.pth`
through the identical split protocol. Note `bn_calibration_source=checkpoint` for epoch 0 means
PyTorch's initial BN buffers (mean 0, var 1), not statistics of any dataset; the NSD-BN row is
the meaningful one for this model.

| Region | BN | Untrained | Default-1000 | CLIP-32 | Trained > untrained |
|---|---|---:|---:|---:|---|
| early | ImageNet | **0.3468** (fc1) | 0.1717 (conv5) | 0.2732 (fc2) | 0/8 for both |
| early | NSD | **0.2917** (conv2) | 0.1532 (conv5) | 0.2452 (conv5) | 0/8 for both |
| ventral | ImageNet | 0.2187 (fc1) | **0.2849** (fc2) | 0.2793 (conv5) | 8/8 for both |
| ventral | NSD | 0.2143 (fc1) | **0.2584** (fc2) | 0.2466 (conv5) | 8/8 for both |

Per-layer selection scores, early cortex, NSD BN:

| Model | conv1 | conv2 | conv3 | conv4 | conv5 | fc1 | fc2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| untrained | −0.204 | 0.244 | 0.242 | 0.230 | 0.237 | 0.221 | 0.223 |
| default-1000 | −0.100 | −0.054 | −0.036 | 0.068 | 0.227 | 0.096 | 0.133 |
| CLIP-32 | −0.096 | −0.041 | −0.004 | 0.072 | 0.276 | 0.246 | 0.196 |

- **In early cortex the untrained network beats both trained networks, in every subject.**
  Its layer profile is flat at ~0.23 from conv2 onward; training drives conv2–conv3 to zero or
  below and leaves conv5 as the only layer near the untrained level. On this stimulus set,
  ImageNet training *removes* early-cortex alignment rather than building it.
- The CLIP-32 > default-1000 early result therefore reads as "coarse training destroys less
  of the initial alignment," not "coarse training builds early-cortex-like features."
  CLIP-32 retains 0.245 of the untrained 0.292; default-1000 retains 0.153.
- **In ventral cortex training helps** (8/8 subjects for both models), and the two trained
  models are indistinguishable there, as before.
- This also explains the conv1–conv3 puzzle from Addendum 3: those layers are not intrinsically
  wrong for early cortex on synthetic stimuli — random conv2/conv3 score 0.24 — they become
  wrong through training.

## Why an untrained network scores high: the RDM is stimulus-type block structure

`split_selection/model_free_baselines.csv`: model-free RDMs scored against each subject's neural
RDM on the same held-out 102 stimuli (Spearman, mean ± SD over 8 subjects).

| Baseline | What it knows | early | ventral |
|---|---|---:|---:|
| coarse family identity | only which of 12 stimulus types (spiral, word, contrast, noise, natscene, …) | **0.284 ± 0.05** | **0.307 ± 0.15** |
| fine family identity | which of 51 variants (spiral_A_sf1, word4_pos2, …) | 0.088 ± 0.01 | 0.083 ± 0.01 |
| raw pixels (after the reference transform) | pixel-space Pearson distance | −0.220 ± 0.05 | −0.069 ± 0.06 |

For comparison, on the same stimuli: untrained 0.292 / 0.214, CLIP-32 0.245 / 0.247,
default-1000 0.153 / 0.258 (early / ventral, NSD BN).

- **A 12-way category indicator matches or beats every network in both regions.** The dominant
  structure in the neural RDM is "spirals respond alike, words respond alike, noise responds
  alike"; a representation that clusters stimuli by type captures most of what RSA can measure
  here. Within-type structure (fine family) contributes only ~0.08.
- **An untrained CNN clusters by type for free.** Random conv filters preserve global image
  statistics (contrast, spatial-frequency energy, texture) that separate the 12 types, so it
  scores at the category-indicator level in early cortex. Raw pixels do not: phase-inverted
  spirals are anti-correlated in pixel space, so pixel distance scrambles the blocks.
- **Trained models fall below the indicator in early cortex** (default-1000 0.15, CLIP-32 0.25
  vs 0.28) and match it in ventral. On this stimulus set, ImageNet training partly erases
  type-level clustering in early layers rather than adding structure beyond it.
- **Consequence:** the synthetic RSA numbers reported above mostly measure how well each model
  preserves stimulus-type clustering, not how well it captures the within-type structure the
  stimuli were designed to probe (spatial frequency, retinal position, contrast). To measure
  the latter, score RSA within each stimulus type, or partial the coarse-family RDM out of both
  model and neural RDMs before comparing.
