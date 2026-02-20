import argparse
from typing import Dict, Any
from base_runner import ExperimentRunner, load_param_grid

BASE_CONFIG = "configs/eval/base.json"


class EvalRunner(ExperimentRunner):
    """Evaluation runner with checkpoint model processing."""

    def __init__(self, base_config, param_grids):
        super().__init__(
            base_config=base_config,
            param_grids=param_grids,
            mode="eval",
            extra_overrides={"log_expdata": True, "load_model_from": "checkpoint"}
        )

    def process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert eval_checkpoint_at_epoch to checkpoint_model and print run info."""
        if "eval_checkpoint_at_epoch" in params:
            epoch = params.pop("eval_checkpoint_at_epoch")
            params["checkpoint_model"] = f"checkpoint_epoch_{epoch}.pth"
            print(f"  Checkpoint: {params['checkpoint_model']}")
            print(f"  Config ID:  cfg{params.get('cfg_id', 'N/A')}")
            print(f"  Results:    results.db")
        return params


def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument("--grid", required=True, help="Parameter grid JSON file")
    args = parser.parse_args()

    runner = EvalRunner(BASE_CONFIG, load_param_grid(args.grid))
    runner.run_all()


if __name__ == "__main__":
    main()
