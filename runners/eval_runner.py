import argparse
from typing import Dict, Any
from base_runner import ExperimentRunner, load_param_grid, console

BASE_CONFIG = "configs/eval/base.json"


class EvalRunner(ExperimentRunner):
    """Evaluation runner with checkpoint model processing."""

    def __init__(self, base_config, param_grids, dry_run=False):
        super().__init__(
            base_config=base_config,
            param_grids=param_grids,
            mode="eval",
            extra_overrides={"log_expdata": True, "load_model_from": "checkpoint"},
            dry_run=dry_run,
        )

    def print_sweep_summary(self, total: int, n_grids: int):
        """Print a high-level summary of what the eval sweep covers."""
        # Collect unique values across all grid groups
        all_datasets, all_analyses = set(), set()
        all_cfg_ids = set()
        n_subjects, n_regions = 0, 0

        for grid in self.param_grids:
            for ds in grid.get("neural_dataset", []):
                all_datasets.add(ds.upper())
            for a in grid.get("analysis", []):
                all_analyses.add(a.upper())
            for c in grid.get("cfg_id", []):
                all_cfg_ids.add(c)
            # Handle nested lists for subject_idx and region
            for s in grid.get("subject_idx", []):
                n_subjects = max(n_subjects, len(s) if isinstance(s, list) else 1)
            for r in grid.get("region", []):
                n_regions = max(n_regions, len(r) if isinstance(r, list) else 1)

        dataset_str = " · ".join(sorted(all_datasets)) or "?"
        analysis_str = " · ".join(sorted(all_analyses)) or "?"
        cfg_ids_sorted = sorted(all_cfg_ids, key=lambda x: (isinstance(x, str), x))
        cfg_str = ", ".join(str(c) for c in cfg_ids_sorted)

        console.print(f"  {dataset_str} · {analysis_str} · {n_subjects} subjects × {n_regions} regions")
        console.print(f"  {total} runs · cfg \\[{cfg_str}]", style="dim")

    def process_params(self, params: Dict[str, Any],
                       run_idx: int, total: int) -> Dict[str, Any]:
        """Convert eval_checkpoint_at_epoch to checkpoint_model and print run info."""
        if "eval_checkpoint_at_epoch" in params:
            epoch = params.pop("eval_checkpoint_at_epoch")
            params["checkpoint_model"] = f"checkpoint_epoch_{epoch}.pth"

        cfg_id = params.get("cfg_id", "?")
        seed = params.get("seed", "?")
        checkpoint_dir = params.get("checkpoint_dir", "")
        folder = checkpoint_dir.rsplit("/", 1)[-1] if checkpoint_dir else "?"

        console.print(
            f"\n  [bold magenta]{run_idx}/{total}[/bold magenta]  "
            f"cfg{cfg_id} · seed {seed} · {folder} "
            f"[dim]· epoch {epoch}[/dim]"
        )
        return params


GRID_DIR = "configs/grids"
VALID_DATASETS = ["nsd", "tvsd", "things", "nsd_synthetic", "cusack"]


def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=VALID_DATASETS, help="Dataset name (resolves to configs/grids/<dataset>.json)")
    group.add_argument("--grid", help="Parameter grid JSON file (explicit path)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    grid_path = f"{GRID_DIR}/{args.dataset}.json" if args.dataset else args.grid
    runner = EvalRunner(BASE_CONFIG, load_param_grid(grid_path), dry_run=args.dry_run)
    runner.run_all()


if __name__ == "__main__":
    main()
