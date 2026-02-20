import argparse
from base_runner import ExperimentRunner, load_param_grid

BASE_CONFIG = "configs/train/base.json"
DEFAULT_GRID = "configs/grids/train_default.json"


def main():
    parser = argparse.ArgumentParser(description="Run training experiments")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="Parameter grid JSON file")
    args = parser.parse_args()

    runner = ExperimentRunner(
        base_config=BASE_CONFIG,
        param_grids=load_param_grid(args.grid),
        mode="train"
    )
    runner.run_all()


if __name__ == "__main__":
    main()
