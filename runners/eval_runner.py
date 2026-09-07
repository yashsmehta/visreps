import argparse
import json
import os
import sqlite3
from typing import Dict, Any, Optional
from base_runner import ExperimentRunner, load_param_grid, console

BASE_CONFIG = "configs/eval/base.json"
RESULTS_DB = "results.db"


class EvalRunner(ExperimentRunner):
    """Evaluation runner with checkpoint model processing."""

    def __init__(self, base_config, param_grids, dry_run=False, skip_done=False):
        super().__init__(
            base_config=base_config,
            param_grids=param_grids,
            mode="eval",
            extra_overrides={"log_expdata": True, "load_model_from": "checkpoint"},
            dry_run=dry_run,
        )
        self.skip_done = skip_done
        with open(self.base_config[0]) as f:
            self.defaults = json.load(f)

    def _get(self, params: Dict[str, Any], key: str, section: Optional[str] = None):
        """Resolve a param, falling back to the base config (optionally a nested section)."""
        if key in params:
            return params[key]
        if section and key in self.defaults.get(section, {}):
            return self.defaults[section][key]
        return self.defaults.get(key)

    def _checkpoint_path(self, params: Dict[str, Any]) -> str:
        """Resolve the checkpoint file this run would load."""
        ckpt_dir = self._get(params, "checkpoint_dir", "checkpoint")
        cfg_id = self._get(params, "cfg_id")
        seed = self._get(params, "seed")
        if "eval_checkpoint_at_epoch" in params:
            model = f"checkpoint_epoch_{params['eval_checkpoint_at_epoch']}.pth"
        else:
            model = self._get(params, "checkpoint_model", "checkpoint")
        seed_letter = chr(ord("a") + int(seed) - 1)
        return f"{ckpt_dir}/cfg{cfg_id}{seed_letter}/{model}"

    def _is_done(self, params: Dict[str, Any]) -> bool:
        """True if results.db already holds every (subject, region) row for this run."""
        if not os.path.exists(RESULTS_DB):
            return False

        def as_list(v):
            return v if isinstance(v, list) else [v]

        dataset = self._get(params, "neural_dataset")
        if dataset == "things-behavior":
            expected = 1  # region/subject_idx are stored as "N/A"
        else:
            expected = len(as_list(self._get(params, "subject_idx"))) * \
                       len(as_list(self._get(params, "region")))

        epoch = params.get("eval_checkpoint_at_epoch")
        with sqlite3.connect(f"file:{RESULTS_DB}?mode=ro", uri=True) as conn:
            n = conn.execute(
                "SELECT COUNT(DISTINCT region || '|' || subject_idx) FROM results "
                "WHERE checkpoint_dir=? AND cfg_id=? AND seed=? AND neural_dataset=? "
                "AND analysis=? AND compare_method=? AND reconstruct_from_pcs=? "
                "AND (? IS NULL OR epoch=?)",
                (
                    self._get(params, "checkpoint_dir", "checkpoint"),
                    self._get(params, "cfg_id"),
                    self._get(params, "seed"),
                    dataset,
                    self._get(params, "analysis"),
                    self._get(params, "compare_method"),
                    int(bool(self._get(params, "reconstruct_from_pcs"))),
                    epoch, epoch,
                ),
            ).fetchone()[0]
        return n >= expected

    def should_skip(self, params: Dict[str, Any]) -> Optional[str]:
        """Skip runs with no checkpoint on disk, and (with --skip-done) ones already evaluated."""
        label = (f"{self._get(params, 'checkpoint_dir', 'checkpoint').rsplit('/', 1)[-1]} "
                 f"cfg{self._get(params, 'cfg_id')} seed {self._get(params, 'seed')} "
                 f"{self._get(params, 'neural_dataset')}")

        path = self._checkpoint_path(params)
        if not os.path.exists(path):
            return f"{label} · no checkpoint at {path}"

        if self.skip_done and self._is_done(params):
            return f"{label} · already in {RESULTS_DB}"

        return None

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
VALID_DATASETS = ["nsd", "tvsd", "things"]


def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=VALID_DATASETS, help="Dataset name (resolves to configs/grids/<dataset>.json)")
    group.add_argument("--grid", help="Parameter grid JSON file (explicit path)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip runs whose results are already in results.db (resume a sweep)")
    args = parser.parse_args()

    grid_path = f"{GRID_DIR}/{args.dataset}.json" if args.dataset else args.grid
    runner = EvalRunner(BASE_CONFIG, load_param_grid(grid_path),
                        dry_run=args.dry_run, skip_done=args.skip_done)
    runner.run_all()


if __name__ == "__main__":
    main()
