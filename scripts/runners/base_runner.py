import json
import subprocess
from itertools import product
from typing import Dict, List, Any, Optional


def load_param_grid(filepath: str) -> Dict[str, List[Any]]:
    """Load parameter grid from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


class ExperimentRunner:
    """Base class for running parameter sweeps over training/evaluation configs."""

    def __init__(
        self,
        base_config: str,
        param_grid: Dict[str, List[Any]],
        mode: str,
        extra_overrides: Optional[Dict[str, Any]] = None
    ):
        self.base_config = base_config
        self.param_grid = param_grid
        self.mode = mode
        self.extra_overrides = extra_overrides or {}

    def run_all(self):
        """Run all parameter combinations in the grid."""
        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_combos = list(product(*self.param_grid.values()))

        total_runs = len(param_combos)
        print(f"Running {total_runs} {self.mode} configurations")

        # Run each combination
        for idx, combo in enumerate(param_combos, 1):
            print(f"\n{'='*60}")
            print(f"Run {idx}/{total_runs} | {(idx/total_runs)*100:.1f}% complete")
            print(f"{'='*60}")

            # Build parameters for this run
            params = dict(zip(param_names, combo))
            params.update(self.extra_overrides)

            # Allow subclasses to modify params
            params = self.process_params(params)

            # Build and run command
            self._run_single(params)

    def process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to modify parameters before execution."""
        return params

    def _run_single(self, params: Dict[str, Any]):
        """Execute a single experiment with given parameters."""
        # Create overrides from parameters (matching original behavior)
        overrides = [f"{k}={json.dumps(v)}" for k, v in params.items()]
        overrides.append(f"mode={self.mode}")

        # Build command
        cmd = [
            "python",
            "-m",
            "visreps.run",
            "--config",
            self.base_config,
            "--override",
        ] + overrides

        cmd_str = " ".join(cmd)
        print(f"\nExecuting: {cmd_str}")
        subprocess.run(cmd)