import json
import subprocess
from itertools import product
from typing import Dict, List, Any, Optional


def load_param_grid(filepath: str) -> List[Dict[str, Any]]:
    """Load parameter grid from JSON file. Expects a JSON array of grid objects."""
    with open(filepath, "r") as f:
        return json.load(f)


class ExperimentRunner:
    """Base class for running parameter sweeps over training/evaluation configs."""

    def __init__(
        self,
        base_config: str,
        param_grids: List[Dict[str, Any]],
        mode: str,
        extra_overrides: Optional[Dict[str, Any]] = None
    ):
        self.base_config = base_config
        self.param_grids = param_grids
        self.mode = mode
        self.extra_overrides = extra_overrides or {}

    def run_all(self):
        """Run all parameter combinations across all grid groups."""
        for grid_idx, param_grid in enumerate(self.param_grids):
            if len(self.param_grids) > 1:
                print(f"\n{'#'*60}")
                print(f"Grid group {grid_idx + 1}/{len(self.param_grids)}")
                print(f"{'#'*60}")
            self._run_grid(param_grid)

    def _run_grid(self, param_grid: Dict[str, Any]):
        """Run all parameter combinations for a single grid group."""
        grid_params = {}
        fixed_params = {}

        for key, value in param_grid.items():
            if isinstance(value, list):
                grid_params[key] = value
            else:
                fixed_params[key] = value

        param_names = list(grid_params.keys())
        param_combos = list(product(*grid_params.values()))

        total_runs = len(param_combos)
        print(f"Running {total_runs} {self.mode} configurations")

        for idx, combo in enumerate(param_combos, 1):
            print(f"\n{'='*60}")
            print(f"Run {idx}/{total_runs} | {(idx/total_runs)*100:.1f}% complete")
            print(f"{'='*60}")

            params = dict(zip(param_names, combo))
            params.update(fixed_params)
            params.update(self.extra_overrides)

            params = self.process_params(params)

            self._run_single(params)

    def process_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to modify parameters before execution."""
        return params

    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dicts into dot-notation overrides."""
        overrides = []
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                overrides.extend(self._flatten_params(value, full_key))
            else:
                # Use json.dumps for proper escaping, but only for complex types
                if isinstance(value, (bool, int, float)):
                    overrides.append(f"{full_key}={json.dumps(value)}")
                elif isinstance(value, str):
                    # Don't double-quote strings
                    overrides.append(f"{full_key}={value}")
                else:
                    overrides.append(f"{full_key}={json.dumps(value)}")
        return overrides

    def _run_single(self, params: Dict[str, Any]):
        """Execute a single experiment with given parameters."""
        # Create overrides from parameters, handling nested dicts
        overrides = self._flatten_params(params)
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