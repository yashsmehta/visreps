from dotenv import load_dotenv
load_dotenv()

import argparse
from visreps.trainer import Trainer
import visreps.evals as evals
import visreps.utils as utils


def main():
    parser = argparse.ArgumentParser(description="Training and evaluation script")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    overrides = list(args.override)
    if args.verbose:
        overrides.append("verbose=true")
    overrides.append(f"mode={args.mode}")
    cfg = utils.load_config(
        args.config or f"configs/{args.mode}/base.json", overrides
    )
    cfg = utils.validate_config(cfg)

    if cfg.mode == "train":
        Trainer(cfg).train()
    else:
        evals.eval(cfg)


if __name__ == "__main__":
    main()
