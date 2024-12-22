import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch

import visreps.trainer as trainer
import visreps.evals as evals


def parse_args():
    parser = argparse.ArgumentParser(description='Training and evaluation script')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval',
                       help='Mode to run: train or eval (default: eval)')
    parser.add_argument('--config', type=str, help='Path to config file. Defaults to configs/{mode}.json')
    parser.add_argument('--override', nargs='*', default=[],
                       help='Override config values, e.g., --override batch_size=32 lr=0.001')
    return parser.parse_args()


def load_config(config_path, override_args):
    """Load config from file and apply CLI overrides"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Apply overrides
    if override_args:
        overrides = OmegaConf.from_dotlist(override_args)
        config = OmegaConf.merge(config, overrides)
    
    return config


def main():
    args = parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = f"configs/{args.mode}.json"
    
    # Load config
    cfg = load_config(args.config, args.override)
    
    # Run requested mode
    if args.mode == "train":
        trainer.train(cfg)
    else:  # eval mode
        evals.eval(cfg)


if __name__ == "__main__":
    main() 