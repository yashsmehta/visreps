# VisReps: At What Level of Categorization Do Neural Networks Capture Ventral Stream Representations?

<p align="left">
<a href="https://docs.python.org/3/whatsnew/3.12.html"><img src="https://img.shields.io/badge/python-3.12-blue.svg?style=for-the-badge&logo=python" alt="Python Version"></a> <a href="https://www.pytorch.com/"><img src="https://img.shields.io/badge/PyTorch-2.5-blue?style=for-the-badge&logo=pytorch&labelColor=gray&color=orange" alt="PyTorch Version"></a> <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative" alt="License"></a>
</p>


## Introduction

Artificial neural networks trained on large-scale object classification tasks exhibit high representational similarity to the human brain. This similarity is typically attributed to training with hundreds or thousands of object categories. In this study, we investigate an alternative question: Can coarse-grained categorization alone achieve similar brain alignment? Using the same dataset (ImageNet/Tiny-ImageNet), we construct broad classification labels based on the principal components of extracted representations from the penultimate layer of a trained AlexNet. We experiment with varying levels of granularity (2, 4, 8, and 16 categories) and analyze how representational similarity analysis (RSA) scores evolve throughout training and across different regions of the ventral stream (early, mid, and higher visual areas) using fMRI responses to natural scenes. Surprisingly, we find that even broad, coarse-grained classification is sufficient to achieve RSA scores comparable to those obtained from networks trained on fine-grained object categories. Additionally, we perform cross-decomposition analysis and further investigate the shared latent dimensions between these networks and the brain. Our findings suggest that high-level ventral stream representations may be driven more by global structure than specific object categories, providing new insights into the nature of neural encoding in artificial and biological vision systems.

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
@software{Mehta_VisReps_2025,
  author = {Mehta, Yash and Bonner, Michael F.},
  title = {{VisReps: At What Level of Categorization Do Neural Networks Capture Ventral Stream Representations?}},
  year = {2025},
  url = {https://github.com/yashsmehta/visreps},
  note = {Accessed: 2025-02-20}
}
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.