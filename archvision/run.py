"""
this file will specify the config, and then call benchmark.py to calculate the brain score metrics.
this will also append the results to a common file.
"""

from omegaconf import OmegaConf
import archvision.evals as evals
import json


if __name__ == "__main__":
    cfg_dict = json.load(open("archvision/configs/default.json"))
    cfg = OmegaConf.create(cfg_dict)

    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)
    evals.eval(cfg)
