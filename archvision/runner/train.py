from omegaconf import OmegaConf
import archvision.trainer as trainer
import json
import sys


def load_config(file_path):
    with open(file_path) as f:
        cfg_dict = json.load(f)
    cli_args = []
    for arg in sys.argv[1:]:
        if '=' in arg and '"' not in arg:
            key, value = arg.split('=')
            if key in ['fc_trainable', 'conv_trainable']:
                cli_args.append(f'{key}="{value}"')
            else:
                cli_args.append(arg)
        else:
            cli_args.append(arg)
    config = OmegaConf.from_cli(cli_args)
    return OmegaConf.merge(OmegaConf.create(cfg_dict), config)

def main():
    config_paths = {
        "train": "archvision/configs/train.json"
    }
    cfg = load_config(config_paths["train"])
    trainer.train(cfg)

if __name__ == "__main__":
    main()

