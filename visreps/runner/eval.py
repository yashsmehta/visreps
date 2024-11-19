from omegaconf import OmegaConf
import visreps.evals as evals
import json


def load_config(file_path):
    """
    Load and merge configuration from a JSON file and command line arguments.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        OmegaConf: A merged OmegaConf object containing settings from both the JSON file and command line arguments.
    """
    with open(file_path) as f:
        cfg_dict = json.load(f)
    config = OmegaConf.from_cli()
    return OmegaConf.merge(OmegaConf.create(cfg_dict), config)


def main():
    config_path = "visreps/configs/eval.json"
    cfg = load_config(config_path)
    evals.eval(cfg)


if __name__ == "__main__":
    main()
