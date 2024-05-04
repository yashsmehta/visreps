from omegaconf import OmegaConf
import archvision.temp_eval as evals
import json

def load_config(file_path):
    with open(file_path) as f:
        cfg_dict = json.load(f)
    config = OmegaConf.from_cli()
    return OmegaConf.merge(OmegaConf.create(cfg_dict), config)

def main():
    config_path = "archvision/configs/temp_eval.json"
    cfg = load_config(config_path)
    evals.eval(cfg)

if __name__ == "__main__":
    main()

