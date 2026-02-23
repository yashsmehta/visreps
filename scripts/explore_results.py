"""
Explore results.db — diagnostic dashboard for evaluation results.

DATABASE SCHEMA (4 tables)
==========================

1. results  (one row per run × compare_method × layer)
   ──────────────────────────────────────────────────────
   Unique constraint: UNIQUE(run_id, compare_method, layer)
   Re-running the same eval replaces old results (INSERT OR REPLACE).

   Identity (NOT NULL):
     run_id          TEXT    — SHA256[:12] of experiment identity fields
     compare_method  TEXT    — "spearman" or "pearson"
     layer           TEXT    — e.g. "conv1"-"conv5", "fc1"-"fc2"

   Scores (nullable):
     score           REAL
     ci_low          REAL    — lower 95% bootstrap CI (NULL if bootstrap=False)
     ci_high         REAL    — upper 95% bootstrap CI

   Experiment (NOT NULL):
     analysis        TEXT    — "rsa" or "encoding_score"
     seed            INTEGER — 1, 2, or 3
     epoch           INTEGER
     neural_dataset  TEXT    — "nsd", "tvsd", "things-behavior", "nsd_synthetic"
     pca_labels      BOOLEAN
     model_name      TEXT    — e.g. "CustomCNN"

   Experiment (nullable):
     region          TEXT    — e.g. "ventral visual stream", "V4", "N/A"
     subject_idx     TEXT    — per-subject index or "N/A" (THINGS)
     cfg_id          INTEGER — 2, 4, 8, 16, 32, 64, or 1000
     pca_n_classes   INTEGER
     pca_labels_folder TEXT  — e.g. "pca_labels_alexnet", "imagenet1k"
     checkpoint_dir  TEXT

   Reconstruction (nullable, with defaults):
     reconstruct_from_pcs BOOLEAN  DEFAULT 0
     pca_k                INTEGER  DEFAULT 1

   Each run_id corresponds to one (subject, region, seed, cfg_id, pca_folder,
   analysis, compare_method, reconstruct_from_pcs, pca_k) combination. The
   results table stores only the best-layer result (1 row per run_id).

2. run_configs  (one row per run_id)
   ──────────────────────────────────
   run_id      TEXT  PRIMARY KEY
   config_json TEXT  NOT NULL — full JSON config used for the run
   created_at  TEXT  DEFAULT datetime('now') — auto-set on insert

   Use JOIN on run_id to recover any training/infra parameter not stored
   directly in `results`.

3. layer_selection_scores  (one row per run × compare_method × layer)
   ──────────────────────────────────────────────────────────────────
   Unique constraint: UNIQUE(run_id, compare_method, layer)

   run_id          TEXT NOT NULL
   compare_method  TEXT NOT NULL
   layer           TEXT NOT NULL
   score           REAL (nullable)

   Stores the per-layer scores used during layer selection (train split).
   Typically 7 rows per run (conv1-5, fc1-2 for CustomCNN). Not all runs
   have layer selection scores (e.g. nsd_synthetic skips selection).

4. bootstrap_distributions  (one row per run × compare_method)
   ──────────────────────────────────────────────────────────
   Unique constraint: UNIQUE(run_id, compare_method)

   run_id          TEXT NOT NULL
   compare_method  TEXT NOT NULL
   scores          TEXT (nullable) — JSON array of 1000 floats

   Raw bootstrap scores (1000 iterations, 90% subsample). Only populated
   when bootstrap=True during eval. Parse with json.loads(scores).

USAGE
=====
    python scripts/explore_results.py                    # full dashboard
    python scripts/explore_results.py --dataset nsd      # filter to one dataset
    python scripts/explore_results.py --dataset nsd --analysis rsa
    python scripts/explore_results.py --query "SELECT * FROM results WHERE cfg_id=1000 LIMIT 5"
"""

import argparse
import os
import sqlite3
import sys
from itertools import groupby
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "results.db"


# ── ANSI colors ──────────────────────────────────────────────────────────────


class _Colors:
    """ANSI escape codes for terminal coloring."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


C = _Colors()
if not sys.stdout.isatty():
    for _attr in ("BOLD", "DIM", "GREEN", "YELLOW", "RED", "CYAN", "MAGENTA", "BLUE", "RESET"):
        setattr(C, _attr, "")

DATASET_COLORS = {
    "nsd": C.CYAN,
    "nsd_synthetic": C.YELLOW,
    "tvsd": C.MAGENTA,
    "things-behavior": C.BLUE,
}

ANALYSIS_COLORS = {
    "rsa": C.CYAN,
    "encoding_score": C.MAGENTA,
}


# ── Expected anatomy per (dataset, analysis) ─────────────────────────────────
# n_subjects defines the expected subject grid.
# Configs with region_groups split regions into subgroups displayed separately.
# Configs with a flat "regions" list show a single completeness table.

EXPECTED_ANATOMY = {
    ("nsd", "rsa"): {
        "n_subjects": 8,
        "region_groups": {
            "Fine-grained ROIs": {
                "regions": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
                "has_reconstruction": False,
            },
            "Streams": {
                "regions": ["early visual stream", "ventral visual stream"],
                "has_reconstruction": True,
            },
        },
    },
    ("nsd", "encoding_score"): {
        "n_subjects": 8,
        "region_groups": {
            "Fine-grained ROIs": {
                "regions": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
                "has_reconstruction": False,
            },
            "Streams": {
                "regions": ["early visual stream", "ventral visual stream"],
                "has_reconstruction": False,
            },
        },
    },
    ("nsd_synthetic", "rsa"): {
        "n_subjects": 8,
        "regions": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
    },
    ("tvsd", "rsa"): {
        "n_subjects": 2,
        "regions": ["V1", "V4", "IT"],
    },
    ("tvsd", "encoding_score"): {
        "n_subjects": 2,
        "regions": ["V1", "V4", "IT"],
    },
    ("things-behavior", "rsa"): {
        "n_subjects": 1,
        "regions": ["N/A"],
    },
}
N_SEEDS = 3


def connect():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────────


def header(title: str):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{'=' * w}")


def dataset_header(ds: str):
    """Bold colored banner for each dataset."""
    color = DATASET_COLORS.get(ds, "")
    w = 70
    print(f"\n{color}{C.BOLD}{'━' * w}")
    print(f"  {ds.upper()}")
    print(f"{'━' * w}{C.RESET}")


def subheader(title: str):
    print(f"\n  {C.BOLD}── {title} {'─' * max(0, 62 - len(title))}{C.RESET}")


def print_df(df: pd.DataFrame, indent: int = 4):
    prefix = " " * indent
    for line in df.to_string(index=False).split("\n"):
        print(f"{prefix}{line}")


def progress_bar(actual: int, expected: int, color: str = "", width: int = 20) -> str:
    """Colored ASCII progress bar. Color represents the analysis type."""
    if expected == 0:
        return f"{C.DIM}{'░' * width}{C.RESET}"
    ratio = min(actual / expected, 1.0)
    filled = round(ratio * width)
    bar_color = color or (C.GREEN if ratio >= 1.0 else C.YELLOW)
    return f"{bar_color}{'█' * filled}{C.RESET}{C.DIM}{'░' * (width - filled)}{C.RESET}"


def _status_str(actual: int, expected: int) -> str:
    """Colored status string."""
    if actual >= expected:
        return f"{C.GREEN}DONE{C.RESET}"
    rem = expected - actual
    if actual == 0:
        return f"{C.RED}{rem} remaining{C.RESET}"
    return f"{C.YELLOW}{rem} remaining{C.RESET}"


# ── Sections ─────────────────────────────────────────────────────────────────


def section_db_info(conn: sqlite3.Connection):
    """File metadata, table sizes, date range."""
    header("DATABASE INFO")
    cur = conn.cursor()

    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"  File:  {DB_PATH}")
    print(f"  Size:  {size_mb:.1f} MB")

    cur.execute("SELECT MIN(created_at), MAX(created_at) FROM run_configs")
    lo, hi = cur.fetchone()
    print(f"  Date range:  {lo}  →  {hi}")

    print()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for (table,) in cur.fetchall():
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table:30s} {count:>8,} rows")


def section_distinct_values(conn: sqlite3.Connection):
    """Unique values for each key column."""
    header("DISTINCT VALUES")
    cur = conn.cursor()
    cols = [
        "neural_dataset",
        "analysis",
        "compare_method",
        "region",
        "cfg_id",
        "seed",
        "pca_labels_folder",
        "model_name",
    ]
    for col in cols:
        cur.execute(f"SELECT DISTINCT {col} FROM results ORDER BY {col}")
        vals = [str(r[0]) for r in cur.fetchall()]
        print(f"  {col:25s} {', '.join(vals)}")


def _show_region_group(
    conn: sqlite3.Connection,
    cur,
    ds: str,
    analysis: str,
    cm: str,
    clause: str,
    n_subj: int,
    group_name: str | None,
    regions: list[str],
    show_reconstruction: bool,
):
    """Show completeness for one group of regions within a (dataset, analysis)."""
    n_reg = len(regions)
    expected_per_config = n_subj * n_reg * N_SEEDS
    acolor = ANALYSIS_COLORS.get(analysis, "")

    # Header
    label = f"{analysis} ({cm})"
    if group_name:
        label += f" -- {group_name}"
    subheader(label)

    print(f"    regions ({n_reg}):  {', '.join(regions)}")
    print(
        f"    expected per config:  "
        f"{n_subj} subj x {n_reg} reg x {N_SEEDS} seeds = {expected_per_config}"
    )

    # Region SQL filter
    region_list = ", ".join(f"'{r}'" for r in regions)
    region_filter = f"AND region IN ({region_list})"

    # Count distinct (subject, region, seed) per (pca_folder, cfg_id)
    df = pd.read_sql(
        f"""
        SELECT
            pca_labels_folder AS pca_folder,
            cfg_id,
            COUNT(DISTINCT subject_idx || '|' || region || '|' || seed) AS actual
        FROM results
        WHERE neural_dataset = '{ds}'
          AND analysis = '{analysis}'
          AND reconstruct_from_pcs = 0
          {region_filter} {clause}
        GROUP BY pca_labels_folder, cfg_id
        ORDER BY pca_labels_folder, cfg_id
        """,
        conn,
    )

    if df.empty:
        print("    (no runs found)")
        return

    df["expected"] = expected_per_config

    # Print table with progress bars
    print()
    for _, row in df.iterrows():
        pf = row["pca_folder"]
        ci = int(row["cfg_id"])
        act = int(row["actual"])
        exp = int(row["expected"])
        bar = progress_bar(act, exp, acolor)
        status = _status_str(act, exp)
        print(f"    {pf:>22s}  {ci:>5d}   {act:>3d}/{exp:<3d}  {bar}  {status}")

    # Overall summary
    total_actual = int(df["actual"].sum())
    total_expected = int(df["expected"].sum())
    total_pct = round(total_actual / total_expected * 100) if total_expected else 0
    n_done = int((df["actual"] >= df["expected"]).sum())
    n_configs = len(df)
    print(
        f"\n    Total: {total_actual}/{total_expected} ({total_pct}%) "
        f"-- {n_done}/{n_configs} configs complete"
    )

    # Reconstruction info (only for groups that have it)
    if show_reconstruction:
        cur.execute(
            f"""
            SELECT COUNT(DISTINCT run_id), COUNT(DISTINCT pca_k)
            FROM results
            WHERE neural_dataset = '{ds}' AND analysis = '{analysis}'
              AND reconstruct_from_pcs = 1 {clause}
            """
        )
        n_recon, n_ks = cur.fetchone()
        if n_recon > 0:
            print(f"    + {n_recon} reconstruction runs ({n_ks} pca_k values)")

    # Baseline check (untrained model: imagenet1k/1000 at epoch 0)
    expected_baseline = n_subj * n_reg * N_SEEDS
    cur.execute(
        f"""
        SELECT COUNT(DISTINCT subject_idx || '|' || region || '|' || seed)
        FROM results
        WHERE neural_dataset = '{ds}' AND analysis = '{analysis}'
          AND pca_labels_folder = 'imagenet1k' AND cfg_id = 1000
          AND epoch = 0 AND reconstruct_from_pcs = 0
          {region_filter} {clause}
        """
    )
    baseline_actual = cur.fetchone()[0]
    bar = progress_bar(baseline_actual, expected_baseline, acolor)
    status = _status_str(baseline_actual, expected_baseline)
    print(
        f"    {'Baseline (epoch 0)':>22s}         "
        f"{baseline_actual:>3d}/{expected_baseline:<3d}  {bar}  {status}"
    )

    # Missing diagnostics (if incomplete)
    if total_actual < total_expected:
        # Per-seed coverage
        cur.execute(
            f"""
            SELECT seed, COUNT(DISTINCT subject_idx || '|' || region) AS n
            FROM results
            WHERE neural_dataset = '{ds}' AND analysis = '{analysis}'
              AND reconstruct_from_pcs = 0 {region_filter} {clause}
            GROUP BY seed ORDER BY seed
            """,
        )
        seed_coverage = cur.fetchall()
        full_per_seed = n_subj * n_reg
        seed_strs = []
        for seed, n in seed_coverage:
            pct = round(n / full_per_seed * 100)
            color = C.GREEN if pct == 100 else C.YELLOW if pct > 0 else C.RED
            seed_strs.append(f"{color}seed {seed}: {n}/{full_per_seed} ({pct}%){C.RESET}")
        missing_seeds = set(range(1, N_SEEDS + 1)) - {s for s, _ in seed_coverage}
        for s in sorted(missing_seeds):
            seed_strs.append(f"{C.RED}seed {s}: 0/{full_per_seed} (0%){C.RESET}")
        print(f"    Seed coverage: {' | '.join(seed_strs)}")

        # Missing regions
        cur.execute(
            f"""
            SELECT DISTINCT region FROM results
            WHERE neural_dataset = '{ds}' AND analysis = '{analysis}'
              AND reconstruct_from_pcs = 0 {region_filter} {clause}
            ORDER BY region
            """,
        )
        present_regions = {r[0] for r in cur.fetchall()}
        missing_regions = set(regions) - present_regions
        if missing_regions:
            print(f"    Missing regions: {', '.join(sorted(missing_regions))}")


def section_completeness(conn: sqlite3.Connection, where: str):
    """For each dataset, show completeness for all analyses.

    Groups output by dataset (with colored headers), then by analysis/region
    group. Uses distinct (subject, region, seed) tuples to avoid inflation
    from multi-epoch or reconstruction sweep runs.
    """
    header("COMPLETENESS")

    clause = f"AND {where}" if where else ""
    cur = conn.cursor()

    # Which (dataset, analysis, compare_method) combos exist?
    cur.execute(
        f"""
        SELECT DISTINCT neural_dataset, analysis, compare_method
        FROM results WHERE 1=1 {clause}
        ORDER BY neural_dataset, analysis
        """
    )
    combos = cur.fetchall()

    # Group by dataset for visual clarity
    for ds, ds_combos in groupby(combos, key=lambda x: x[0]):
        dataset_header(ds)

        for _, analysis, cm in ds_combos:
            anatomy = EXPECTED_ANATOMY.get((ds, analysis))
            if not anatomy:
                subheader(f"{analysis} ({cm})  [no expected anatomy defined]")
                continue

            n_subj = anatomy["n_subjects"]

            if "region_groups" in anatomy:
                for group_name, group_info in anatomy["region_groups"].items():
                    _show_region_group(
                        conn, cur, ds, analysis, cm, clause,
                        n_subj, group_name, group_info["regions"],
                        group_info.get("has_reconstruction", False),
                    )
            else:
                _show_region_group(
                    conn, cur, ds, analysis, cm, clause,
                    n_subj, None, anatomy["regions"],
                    show_reconstruction=True,
                )


def section_health(conn: sqlite3.Connection):
    """Integrity checks."""
    header("HEALTH CHECKS")
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(DISTINCT r.run_id) FROM results r "
        "LEFT JOIN run_configs rc ON r.run_id = rc.run_id "
        "WHERE rc.run_id IS NULL"
    )
    orphaned = cur.fetchone()[0]
    status = f"{C.GREEN}OK{C.RESET}" if orphaned == 0 else f"{C.RED}WARN: {orphaned} orphaned{C.RESET}"
    print(f"  run_configs coverage:       {status}")

    total = pd.read_sql("SELECT COUNT(DISTINCT run_id) AS n FROM results", conn).iloc[0, 0]

    cur.execute(
        "SELECT COUNT(DISTINCT r.run_id) FROM results r "
        "LEFT JOIN bootstrap_distributions bd "
        "  ON r.run_id = bd.run_id AND r.compare_method = bd.compare_method "
        "WHERE bd.run_id IS NULL"
    )
    no_boot = cur.fetchone()[0]
    if no_boot == 0:
        print(f"  bootstrap_distributions:    {C.GREEN}OK{C.RESET} ({total}/{total})")
    else:
        print(f"  bootstrap_distributions:    {C.YELLOW}{total - no_boot}/{total}{C.RESET} have bootstrap")

    cur.execute(
        "SELECT COUNT(DISTINCT r.run_id) FROM results r "
        "LEFT JOIN (SELECT DISTINCT run_id FROM layer_selection_scores) ls "
        "  ON r.run_id = ls.run_id WHERE ls.run_id IS NULL"
    )
    no_ls = cur.fetchone()[0]
    if no_ls == 0:
        print(f"  layer_selection_scores:     {C.GREEN}OK{C.RESET} ({total}/{total})")
    else:
        print(f"  layer_selection_scores:     {C.YELLOW}{total - no_ls}/{total}{C.RESET} have layer selection")
        # Break down why — reconstruction sweep vs nsd_synthetic vs unexpected
        df_ls = pd.read_sql(
            """
            SELECT r.neural_dataset, r.reconstruct_from_pcs AS recon,
                   COUNT(DISTINCT r.run_id) AS missing
            FROM results r
            LEFT JOIN (SELECT DISTINCT run_id FROM layer_selection_scores) ls
              ON r.run_id = ls.run_id
            WHERE ls.run_id IS NULL
            GROUP BY r.neural_dataset, r.reconstruct_from_pcs
            ORDER BY r.neural_dataset, r.reconstruct_from_pcs
            """,
            conn,
        )
        recon_missing = df_ls[df_ls["recon"] == 1]["missing"].sum()
        baseline_missing = df_ls[df_ls["recon"] == 0]["missing"].sum()
        print(f"    reconstruction sweep (expected):  {recon_missing}")
        if baseline_missing > 0:
            baseline_rows = df_ls[df_ls["recon"] == 0]
            datasets = ", ".join(baseline_rows["neural_dataset"].tolist())
            print(
                f"    baseline runs (reuses parent):    {baseline_missing}  ({datasets})"
            )

    cur.execute("SELECT COUNT(*) FROM results WHERE score IS NULL")
    null_scores = cur.fetchone()[0]
    status = f"{C.GREEN}OK{C.RESET}" if null_scores == 0 else f"{C.RED}WARN: {null_scores} rows{C.RESET}"
    print(f"  NULL scores:                {status}")


def section_recent(conn: sqlite3.Connection, n: int):
    """Most recently added runs."""
    header(f"RECENT RUNS  (last {n})")
    df = pd.read_sql(
        f"""
        SELECT rc.created_at, r.neural_dataset, r.analysis,
               r.pca_labels_folder, r.cfg_id, r.seed, r.region, r.subject_idx
        FROM run_configs rc
        JOIN results r ON rc.run_id = r.run_id
        ORDER BY rc.created_at DESC
        LIMIT {n}
        """,
        conn,
    )
    print_df(df)


def run_custom_query(conn: sqlite3.Connection, query: str):
    """Run an arbitrary SELECT query."""
    header("CUSTOM QUERY")
    print(f"  {query}\n")
    df = pd.read_sql(query, conn)
    with pd.option_context("display.max_rows", 60, "display.width", 160):
        print_df(df)
    print(f"\n  ({len(df)} rows returned)")


# ── Main ─────────────────────────────────────────────────────────────────────


def build_where(args) -> str:
    clauses = []
    if args.dataset:
        clauses.append(f"neural_dataset = '{args.dataset}'")
    if args.analysis:
        clauses.append(f"analysis = '{args.analysis}'")
    if args.region:
        clauses.append(f"region = '{args.region}'")
    if args.cfg_id is not None:
        clauses.append(f"cfg_id = {args.cfg_id}")
    if args.compare_method:
        clauses.append(f"compare_method = '{args.compare_method}'")
    return " AND ".join(clauses)


def main():
    parser = argparse.ArgumentParser(
        description="Explore results.db — diagnostic dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, help="Filter by neural_dataset")
    parser.add_argument("--analysis", type=str, help="Filter by analysis type")
    parser.add_argument("--region", type=str, help="Filter by region")
    parser.add_argument("--cfg_id", type=int, help="Filter by cfg_id")
    parser.add_argument("--compare_method", type=str, help="Filter by compare_method")
    parser.add_argument("--query", type=str, help="Run a custom SQL SELECT query")
    parser.add_argument(
        "--recent", type=int, default=10, help="Number of recent runs to show"
    )
    args = parser.parse_args()

    conn = connect()

    if args.query:
        run_custom_query(conn, args.query)
        conn.close()
        return

    where = build_where(args)
    if where:
        print(f"\n  Active filters: {where}")

    section_db_info(conn)
    section_distinct_values(conn)
    section_completeness(conn, where)
    section_health(conn)
    section_recent(conn, args.recent)

    conn.close()


if __name__ == "__main__":
    main()
