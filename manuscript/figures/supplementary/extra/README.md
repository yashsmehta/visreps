# Supplementary Figures

**Parent document:** `manuscript/figures/paper.md`

All supplementary figures are generated from `manuscript/figures/supplementary/` scripts. Run each script from the **project root** (`visreps/`).

---

## Figure Index

Figures are organized thematically: validation → extended main results → anatomical detail → robustness → alternative labels/metrics → representational analysis → behavioral deep-dives → interpretability → external benchmarks.

| Figure | Title | Script | Key Takeaway |
|--------|-------|--------|-------------|
| **S1** | Training Summary & Model Accuracy | `supp_s1_training_summary.py` | All models converge; coarse models reach near-perfect accuracy on their respective tasks. |
| **S2** | Summary Bar Comparison | `supp_s2_summary_bars.py` | Cross-dataset bar comparison: pretrained (ViT, CLIP) vs. trained from scratch (untrained, 1K, best coarse). |
| **S3** | Full Per-Layer Profiles (All 7 Levels) | `supp_s3_full_per_layer.py` | Complete per-layer profiles for all granularity levels (2–1000) across all datasets. |
| **S4** | Neural Reconstruction Analysis | `supp_s4_neural_reconstruction.py` | Reconstruction control: alignment vs. PCs retained for TVSD (V1, V4, IT) and NSD (early, ventral). |
| **S5** | THINGS Per-Architecture Breakdown | `supp_s5_things_architectures.py` | Individual coarseness curves for each PCA source on THINGS behavioral benchmark. |
| **S6** | Fine-Grained ROI Analysis (NSD) | `supp_s6_finegrained_roi.py` | Coarseness effect at finer anatomical resolution (V1, V2, V3, hV4, FFA, PPA). |
| **S7** | NSD-Synthetic (OOD) Results | `supp_s7_nsd_synthetic.py` | Coarse model alignment holds on out-of-distribution synthetic stimuli. |
| **S8** | Stimulus Robustness | `supp_s8_stimulus_robustness.py` | RSA stability under stimulus subsampling. |
| **S9** | Score Distributions | `supp_s9_score_distributions.py` | Violin plots showing full spread of scores across subjects × seeds per condition. |
| **S10** | Additional PCA Source Models (ViT + DINOv3) | `supp_s10_dinov2.py` | Main findings replicate with ViT and DINOv3 labels — not contingent on any single PCA source. |
| **S11** | WordNet Hierarchy | `supp_s11_wordnet.py` | Results using WordNet-derived coarse labels as an alternative to PCA-based labels. |
| **S12** | Internal Representation Analysis | `supp_s12_representation_summary.py` | Eigenspectrum, participation ratio, intrinsic dimensionality, and sparsity across granularity. |
| ~~**S13**~~ | *(Moved to Figure 5C)* | `manuscript/figures/fig5/dimension_profiling.py` | — |
| **S14** | Image Collages | `supp_s14_image_collages.py` | Representative images for concepts where coarse wins vs. 1000-way wins. |
| **S15** | PC Axis Interpretation | `supp_s15_pc_poles.py` | Most/least activating images for the top PCs of each source model. |
| **S16** | Levels Evaluation | `supp_s16_levels.py` | Results on the hierarchical Levels benchmark (Muttenthaler et al. 2025). |
| **S17** | Seed Variability | `supp_s17_seed_variability.py` | Score variability across seeds. |
| **S18** | Class-Level RDM Grid | `plot_class_rdms.py` | Class-level RDMs across all granularity levels (2,4,8,16,32,64,1000-way). Moved from old main Figure 2A; supplementary version includes all levels. |

---

## Running All Figures

```bash
# From project root (visreps/)
source .venv/bin/activate

# DB-only figures (no GPU needed)
python manuscript/figures/supplementary/supp_s1_training_summary.py
python manuscript/figures/supplementary/supp_s2_summary_bars.py
python manuscript/figures/supplementary/supp_s3_full_per_layer.py
python manuscript/figures/supplementary/supp_s4_neural_reconstruction.py
python manuscript/figures/supplementary/supp_s5_things_architectures.py
python manuscript/figures/supplementary/supp_s6_finegrained_roi.py
python manuscript/figures/supplementary/supp_s7_nsd_synthetic.py
python manuscript/figures/supplementary/supp_s8_stimulus_robustness.py
python manuscript/figures/supplementary/supp_s9_score_distributions.py
python manuscript/figures/supplementary/supp_s10_dinov2.py
python manuscript/figures/supplementary/supp_s11_wordnet.py
python manuscript/figures/supplementary/supp_s12_representation_summary.py
# S13 moved to main Figure 5C: python manuscript/figures/fig5/dimension_profiling.py
python manuscript/figures/supplementary/supp_s16_levels.py
python manuscript/figures/supplementary/supp_s17_seed_variability.py
python manuscript/figures/supplementary/plot_class_rdms.py

# Image-loading figures (need ImageNet access)
python manuscript/figures/supplementary/supp_s14_image_collages.py
python manuscript/figures/supplementary/supp_s15_pc_poles.py
```

---

## Data Sources

| Source | Location | Used By |
|--------|----------|---------|
| Results DB | `results.db` | S2–S7, S9–S11 |
| Training metrics | `/data/ymehta3/{alexnet_pca,default}/cfg*/training_metrics.csv` | S1 |
| Stimulus robustness cache | `manuscript/figures/supplementary/supp_s8_data.npz` | S8 |
| Representation analysis | `experiments/representation_analysis/figs/*.json` | S12 |
| THINGS viz data | `experiments/things_visualizations/data/things_viz_data.npz` | Fig 5C, S14 |
| Levels evaluation | `experiments/levels_evaluation/levels_summary.csv` | S16 |
| PCA poles | `datasets/obj_cls/imagenet/pca_poles/` | S15 |

## Color Scheme

All figures use the shared color scheme from `manuscript/figures/fig_utils.py`:
- **PCA sources (coarseness):** AlexNet (teal `#1a9e76`), CLIP (purple `#7b3294`), ViT (crimson `#d62728`), Pixels (brown `#8c564b`)
- **Granularity (per-layer):** Blue gradient (2=`#c6dbef` → 64=`#084594`), 1000-way=`#e6550d`
- **Baselines:** Untrained (gray `#AAAAAA` dashed), 1000-way (orange-red `#e6550d`)
