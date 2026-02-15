# Probing the Granularity of Human-Machine Alignment

<p align="left">
<a href="https://docs.python.org/3/whatsnew/3.11.html"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python" alt="Python Version"></a> <a href="https://www.pytorch.com/"><img src="https://img.shields.io/badge/PyTorch-2.0+-blue?style=for-the-badge&logo=pytorch&labelColor=gray&color=orange" alt="PyTorch Version"></a> <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative" alt="License"></a>
</p>

## Introduction

Deep neural networks for object classification align closely with human visual representations, a correspondence that has been attributed to fine-grained category supervision. We investigate whether such granular supervision is necessary for robust brain-model alignment. Using a PCA-based method, we generate progressively coarser ImageNet label sets (ranging from 2 to 64 categories) and retrain a standard CNN (AlexNet) from scratch for each granularity, enabling controlled comparisons against standard 1000-class training.

Evaluations employ representational similarity analysis (RSA) on large-scale fMRI data (NSD, including out-of-distribution stimuli) and behavioral data (THINGS). Our key findings include:
1. On behavioral data, models trained with minimal categories (e.g., 2 classes) achieve surprisingly high alignment with human similarity judgments
2. On fMRI data, models trained with 32-64 categories match or outperform 1000-class models in early visual cortex alignment and exhibit comparable performance in ventral areas, with coarser models displaying advantages on OOD stimuli
3. Coarse-trained representations differ structurally from low-dimensional projections of fine-grained models, suggesting the learning of novel visual features

Collectively, these findings indicate that broader categorical distinctions are often sufficient — and sometimes more effective — for capturing cognitively salient visual structure, especially in early visual processing and OOD contexts. This work introduces classification granularity as a new framework for probing visual representation alignment, laying the groundwork for more biologically-aligned vision systems.

## Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:yashsmehta/visreps.git
   cd visreps
   ```

2. **Set Up Python Environment** (requires Python 3.11+)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   source .venv/bin/activate
   ```

3. **Configure Environment**
   Copy the example environment file and fill in paths to your datasets:
   ```bash
   cp .env.example .env
   # Edit .env with your dataset paths
   ```

## Usage

### Training
Train models with different label granularities:
```bash
# Train with 32 PCA-derived classes
python -m visreps.run --mode train --override pca_labels=true pca_n_classes=32 seed=1

# Grid search over multiple configurations
python scripts/runners/train_runner.py --grid configs/grids/train_default.json
```

### Evaluation
Evaluate brain-model alignment:
```bash
# RSA on NSD fMRI data
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=rsa neural_dataset=nsd

# Encoding score on THINGS behavioral data
python -m visreps.run --mode eval --override cfg_id=32 seed=1 analysis=encoding_score neural_dataset=things

# Grid search over evaluation configurations
python scripts/runners/eval_runner.py --grid configs/grids/eval_default.json
```

Configuration files are in `configs/train/` and `configs/eval/`. Use `--override` to modify parameters from the command line.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{GranularityAlignment2025,
  author = {Author Names},
  title = {{Probing the Granularity of Human-Machine Alignment}},
  year = {2025},
  url = {https://github.com/yashsmehta/visreps},
}
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
