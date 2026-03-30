import json
import subprocess
from itertools import product
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "info": "bold white",
    "success": "green",
    "highlight": "bold magenta",
    "dim": "dim",
})
console = Console(theme=_theme)


def load_param_grid(filepath: str) -> List[Dict[str, Any]]:
    """Load parameter grid from JSON file. Expects a JSON array of grid objects."""
    with open(filepath, "r") as f:
        return json.load(f)


class ExperimentRunner:
    """Base class for running parameter sweeps over training/evaluation configs."""

    def __init__(
        self,
        base_config,
        param_grids: List[Dict[str, Any]],
        mode: str,
        extra_overrides: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ):
        # Accept a single path or list of paths
        if isinstance(base_config, str):
            self.base_config = [base_config]
        else:
            self.base_config = list(base_config)
        self.param_grids = param_grids
        self.mode = mode
        self.extra_overrides = extra_overrides or {}
        self.dry_run = dry_run

    def _total_runs(self):
        """Count total runs across all grid groups."""
        total = 0
        for param_grid in self.param_grids:
            grid_params = {k: v for k, v in param_grid.items() if isinstance(v, list)}
            total += len(list(product(*grid_params.values()))) if grid_params else 1
        return total

    def run_all(self):
        """Run all parameter combinations across all grid groups."""
        total = self._total_runs()
        n_grids = len(self.param_grids)

        console.print(f"\n  ── {self.mode.upper()} sweep {'─' * 38}", style="info")
        self.print_sweep_summary(total, n_grids)

        global_idx = 0
        for grid_idx, param_grid in enumerate(self.param_grids):
            if n_grids > 1:
                console.print(f"\n  ── Grid {grid_idx + 1}/{n_grids} {'─' * 40}", style="info")
            global_idx = self._run_grid(param_grid, global_idx, total)

        console.print(f"\n  ── Complete {'─' * 41}", style="success")

    def _run_grid(self, param_grid: Dict[str, Any], global_idx: int, total: int):
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

        for combo in param_combos:
            global_idx += 1

            params = dict(zip(param_names, combo))
            params.update(fixed_params)
            params.update(self.extra_overrides)

            params = self.process_params(params, global_idx, total)

            self._run_single(params)

        return global_idx

    def print_sweep_summary(self, total: int, n_grids: int):
        """Hook for subclasses to print a sweep-level summary."""
        console.print(f"  {total} runs across {n_grids} grid group{'s' if n_grids > 1 else ''}", style="dim")

    def process_params(self, params: Dict[str, Any],
                       run_idx: int, total: int) -> Dict[str, Any]:
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

        # Build command — pass mode as a CLI flag (not an override)
        # so it isn't clobbered by run.py's default mode=eval append
        cmd = [
            "python",
            "-m",
            "visreps.run",
            "--mode",
            self.mode,
            "--config",
        ] + self.base_config + [
            "--override",
        ] + overrides

        if not self.dry_run:
            subprocess.run(cmd)
