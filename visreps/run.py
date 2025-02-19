import argparse
from visreps.trainer import Trainer
import visreps.evals as evals
import visreps.utils as utils


def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", nargs="*", default=[])

    args = parser.parse_args()
    cfg = utils.load_config(
        args.config or f"configs/{args.mode}/base.json", args.override
    )
    cfg = utils.validate_config(cfg)

    if cfg.mode == "train":
        trainer = Trainer(cfg)
        trainer.train()
    else:
        evals.eval(cfg)


if __name__ == "__main__":
    main()
