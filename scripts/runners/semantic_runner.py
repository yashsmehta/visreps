import argparse
import subprocess
from typing import Dict, Any
from base_runner import ExperimentRunner, load_param_grid

BASE_CONFIG = "configs/eval/base.json"
DEFAULT_GRID = "configs/grids/semantic_align.json"
GEMINI_FEATURES_PATH = "datasets/neural/nsd/gemini_representations.npz"


class SemanticRunner(ExperimentRunner):
    """Semantic alignment runner using Gemini embeddings."""

    def __init__(self, base_config, param_grid):
        super().__init__(
            base_config=base_config,
            param_grid=param_grid,
            mode="semantic",
            extra_overrides={
                "log_expdata": True,
                "load_model_from": "checkpoint",
                "gemini_features_path": GEMINI_FEATURES_PATH
            }
        )

    def process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert eval_checkpoint_at_epoch to checkpoint_model and print run info."""
        if "eval_checkpoint_at_epoch" in params:
            epoch = params.pop("eval_checkpoint_at_epoch")
            params["checkpoint_model"] = f"checkpoint_epoch_{epoch}.pth"
            print(f"  Checkpoint: {params['checkpoint_model']}")
            print(f"  Config ID:  cfg{params.get('cfg_id', 'N/A')}")
            print(f"  Results:    {params.get('results_csv', 'N/A')}")
        return params

    def _run_single(self, params: Dict[str, Any]):
        """Execute semantic alignment with given parameters."""
        overrides = self._flatten_params(params)
        
        cmd = [
            "python",
            "experiments/semantic_alignment.py",
            "--config",
            self.base_config,
            "--override",
        ] + overrides

        cmd_str = " ".join(cmd)
        print(f"\nExecuting: {cmd_str}")
        subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run semantic alignment experiments")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="Parameter grid JSON file")
    args = parser.parse_args()

    runner = SemanticRunner(BASE_CONFIG, load_param_grid(args.grid))
    runner.run_all()


if __name__ == "__main__":
    main()

