# Experiments

Standalone analysis scripts. Each subdirectory is self-contained with its own
analysis, plotting, and data. Run all scripts from the project root.

| Directory | Purpose | Key outputs |
|-----------|---------|-------------|
| `neurips_2025/` | NeurIPS submission figures (fig1-fig4) | Publication PNGs |
| `1k_pretrained/` | RSA of pretrained models (AlexNet, ViT) on NSD | `logs/1k_pretrained_nsd_rsa.csv` |
| `binary_pc_rsa/` | RSA using binary PC splits vs CNN baselines | `binary_pc_rsa.csv` |
| `coarse_grain_benefits/` | Downstream benefits: few-shot, robustness, curriculum, linear probe | `results/*.csv` |
| `pca_analysis/` | PCA label structure and class distribution visualization | PNGs |
| `representation_analysis/` | Dimensionality, nearest neighbors, variance, task-brain alignment | `figs/`, `dimensionality/` |
| `semantic_analysis/` | Semantic structure: WordNet, UMAP, Gemini alignment | PC histograms, UMAP PNGs |
| `wordnet/` | WordNet hierarchy utilities and label generation | `semantic_categories.csv` |
| `stimulus_robustness/` | Verify coarse > fine alignment is robust to stimulus subsampling | `data.json`, `stimulus_robustness.png` |
| `stimulus_sensitivity/` | k-fold CV RSA fluctuation analysis | `data.json`, `stimulus_sensitivity.png` |

Note: Some scripts combine computation and plotting in one file.
