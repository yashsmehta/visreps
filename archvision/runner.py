"""
this file will specify the config, and then call benchmark.py to calculate the brain score metrics.
this will also append the results to a common file.
"""

from omegaconf import OmegaConf
import archvision.evals as evals


if __name__ == "__main__":

    cfg_dict = {
    "wavelet": {
        "layers": 1,
        "type": "gabor",
    },
    "conv": {
        "layers": 4,
        "layer_sizes": [64, 128, 256, 256],
        "kernel_sizes": [7, 5, 3, 3],
        "init": "kaiming",
        "norm": "layer",
        "nonlin": "relu",
        },
    "voxel_set": "OTC",
    "eval_metric": "srpr",
    "benchmark_data_type": "fMRI",
    "log_expdata": True,  # flag to save the training data
    "log_dir": "logs/",  # directory to save experimental data
    "exp_name": "test",  # logs are stored under a directory created under this name
    }

    cfg = OmegaConf.create(cfg_dict)

    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)
    evals.run(cfg)
