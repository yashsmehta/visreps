import argparse
from base_runner import ExperimentRunner, load_param_grid

BASE_CONFIG = "configs/train/base_local.json"
DEFAULT_GRID = "configs/grids/train_default.json"


def main():
    parser = argparse.ArgumentParser(description="Run training experiments (local)")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="Parameter grid JSON file")
    parser.add_argument("--arch", required=True,
                        help="Architecture config (e.g. configs/train/architectures/custom_cnn.json)")
    parser.add_argument("--base", default=BASE_CONFIG, help="Base config file")
    args = parser.parse_args()

    runner = ExperimentRunner(
        base_config=[args.base, args.arch],
        param_grids=load_param_grid(args.grid),
        mode="train"
    )
    runner.run_all()


if __name__ == "__main__":
    main()
