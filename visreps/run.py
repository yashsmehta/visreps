import argparse
from pathlib import Path
from omegaconf import OmegaConf

from visreps.trainer import Trainer
import visreps.evals as evals


def parse_args():
    parser = argparse.ArgumentParser(description='Training and evaluation script')
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval',
                        help='Whether to train or eval (default: eval)')
    parser.add_argument('--config', default=None,
                        help='Path to config file. Defaults to configs/{mode}/base.json')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config values, e.g. batch_size=32 lr=0.001')

    args = parser.parse_args()
    # Set default config path
    if not args.config:
        args.config = f"configs/{args.mode}/base.json"
    return args


def load_config(config_path, overrides):
    """Load config from file and apply CLI overrides."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Mode-specific configs
    if "eval" in config_path:
        model_paths = {
            "torchvision": "configs/eval/torchvision_model.json",
            "checkpoint":  "configs/eval/checkpoint_model.json"
        }
        model_cfg_path = model_paths.get(cfg.load_model_from)
        if model_cfg_path:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(model_cfg_path))
    elif "train" in config_path:
        model_paths = {
            "standard_cnn": "configs/train/standard_cnn.json",
            "custom_cnn":   "configs/train/custom_cnn.json"
        }
        model_cfg_path = model_paths.get(cfg.model_class)
        if model_cfg_path:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(model_cfg_path))

    # Apply CLI overrides
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)

    if args.mode == "train":
        trainer = Trainer(cfg)
        trainer.train()
    else:
        evals.eval(cfg)


if __name__ == "__main__":
    main()