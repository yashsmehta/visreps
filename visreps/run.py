from omegaconf import OmegaConf
import visreps.trainer as trainer
import visreps.evals as evals
import json
import sys


def load_config(file_path):
    """
    Load and merge configuration from a JSON file and command line arguments.

    This function reads a configuration from a JSON file specified by `file_path`,
    then merges it with configurations provided via command line arguments. Command
    line arguments are expected in the format `key=value`. If the key is one of the
    special cases ('fc_trainable', 'conv_trainable'), it is cast to string
    before merging.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        OmegaConf.DictConfig: A merged configuration object.
    """
    with open(file_path) as f:
        cfg_dict = json.load(f)
    cli_args = []
    for arg in sys.argv[1:]:
        if "=" in arg and '"' not in arg:
            key, value = arg.split("=")
            if key in ["fc_trainable", "conv_trainable"]:
                cli_args.append(f'{key}="{value}"')
            else:
                cli_args.append(arg)
        else:
            cli_args.append(arg)
    config = OmegaConf.from_cli(cli_args)
    return OmegaConf.merge(OmegaConf.create(cfg_dict), config)


def main():
    # Define config paths for different modes
    config_paths = {
        "train": "visreps/configs/train.json",
        "eval": "visreps/configs/eval.json"
    }

    # Get mode from command line arguments, default to train
    mode = "train"  # default mode
    for arg in sys.argv[1:]:
        if arg.startswith("mode="):
            mode = arg.split("=")[1]
            break

    if mode not in config_paths:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(config_paths.keys())}")

    # Load config and run appropriate function
    cfg = load_config(config_paths[mode])
    
    if mode == "train":
        trainer.train(cfg)
    elif mode == "eval":
        evals.eval(cfg)


if __name__ == "__main__":
    main() 