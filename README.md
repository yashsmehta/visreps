# VisReps: How do cortex-like representations emerge in deep neural networks during training?

<p align="left">
<a href="https://docs.python.org/3/whatsnew/3.12.html"><img src="https://img.shields.io/badge/python-3.12-blue.svg?style=for-the-badge&logo=python" alt="Python Version"></a> <a href="https://www.pytorch.com/"><img src="https://img.shields.io/badge/PyTorch-2.5-blue?style=for-the-badge&logo=pytorch&labelColor=gray&color=orange" alt="PyTorch Version"></a> <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative" alt="License"></a>
</p>


## Introduction

How do neural representations in the human visual cortex and deep neural networks become similar during training? This repository contains code to explore how representational similarities between artificial neural networks and the human visual system develop over time. It includes a training pipeline (focused on object classification tasks, using standard CNNs or customizable CNNs with configurable trainable layers) and an evaluation pipeline (offering metrics like RSA, encoding scores, and cross-correlation to measure similarity to the visual cortex).

## Installation

1. **Clone the Repository**
   ```bash
   git clone git@github.com:yashsmehta/visreps.git
   cd visreps
   ```

2. **Configure Environment**
   Edit `.env` file in the project root:
   ```bash
   PROJECT_HOME="/path/to/visreps"
   ```

3. **Set Up Python Environment**
   Install uv package manager and create a virtual environment:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   source .venv/bin/activate
   ```

## Usage

The code is organized into two main components: training and evaluation. The common entry point is run.py, which uses configuration files located in the configs folder. Simply modify or specify the desired configuration file to run your experiments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{Mehta_VisReps_2024,
  author = {Mehta, Yash},
  title = {{VisReps: How do cortex-like representations emerge in deep neural networks during training?}},
  year = {2024},
  url = {https://github.com/yashsmehta/visreps},
  note = {Accessed: 2024-10-01}
}
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.