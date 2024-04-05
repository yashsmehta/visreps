"""
this file will specify the config, and then call benchmark.py to calculate the brain score metrics.
this will also append the results to a common file.
"""

from omegaconf import OmegaConf
import archvision.evals as evals


if __name__ == "__main__":

    cfg_dict = {
        "voxel_set": "OTC",  # region of interest in the data
        "eval_metric": "ersa",
        "benchmark_data_type": "fMRI",
        "model_uid": "torchvision_alexnet_imagenet1k_v1",
        "log_expdata": True,  # flag to save the training data
        "log_dir": "logs/",  # directory to save experimental data
        "exp_name": "test",  # logs are stored under a directory created under this name
    }

    cfg = OmegaConf.create(cfg_dict)

    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)
    evals.run(cfg)
