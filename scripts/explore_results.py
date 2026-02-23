"""Explore results.db -- diagnostic dashboard for evaluation results.

First SEE CLAUDE.md (project root) for full DB schema (4 tables: results,
run_configs, layer_selection_scores, bootstrap_distributions).

Usage:
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

# -- ANSI colors ---------------------------------------------------------------

_CODES = dict(BOLD=1, DIM=2, RED=31, GREEN=32, YELLOW=33, BLUE=34, MAGENTA=35, CYAN=36)
_use_color = sys.stdout.isatty()

class _C:
    RESET = "\033[0m" if _use_color else ""
for _name, _code in _CODES.items():
    setattr(_C, _name, f"\033[{_code}m" if _use_color else "")

C = _C()

DATASET_COLORS = {"nsd": C.CYAN, "nsd_synthetic": C.YELLOW, "tvsd": C.MAGENTA, "things-behavior": C.BLUE}
ANALYSIS_COLORS = {"rsa": C.CYAN, "encoding_score": C.MAGENTA}


# -- Expected anatomy per (dataset, analysis) ----------------------------------
# Each entry: n_subjects, groups (list of dicts with name/regions/has_reconstruction).
# "name" is None for single-group datasets; non-None adds "-- Name" to the header.

def _grp(regions, *, name=None, has_reconstruction=False):
    return {"name": name, "regions": regions, "has_reconstruction": has_reconstruction}

_NSD_FINE = _grp(["V1", "V2", "V3", "hV4", "FFA", "PPA"], name="Fine-grained ROIs")
_NSD_FINE_RECON = _grp(["V1", "V2", "V3", "hV4", "FFA", "PPA"], name="Fine-grained ROIs")
_NSD_STREAMS = _grp(["early visual stream", "ventral visual stream"], name="Streams", has_reconstruction=True)
_NSD_STREAMS_NO_RECON = _grp(["early visual stream", "ventral visual stream"], name="Streams")

EXPECTED_ANATOMY = {
    ("nsd", "rsa"):            {"n_subjects": 8, "groups": [_NSD_FINE, _NSD_STREAMS]},
    ("nsd", "encoding_score"): {"n_subjects": 8, "groups": [_NSD_FINE, _NSD_STREAMS_NO_RECON]},
    ("nsd_synthetic", "rsa"):  {"n_subjects": 8, "groups": [_grp(["V1", "V2", "V3", "hV4", "FFA", "PPA"], name="Fine-grained ROIs", has_reconstruction=True), _NSD_STREAMS]},
    ("tvsd", "rsa"):           {"n_subjects": 2, "groups": [_grp(["V1", "V4", "IT"], has_reconstruction=True)]},
    ("tvsd", "encoding_score"):{"n_subjects": 2, "groups": [_grp(["V1", "V4", "IT"], has_reconstruction=True)]},
    ("things-behavior", "rsa"):{"n_subjects": 1, "groups": [_grp(["N/A"], has_reconstruction=True)]},
}
N_SEEDS = 3


def connect():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


# -- Helpers --------------------------------------------------------------------

def header(title: str):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{'=' * w}")


def dataset_header(ds: str):
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
    if expected == 0:
        return f"{C.DIM}{'░' * width}{C.RESET}"
    ratio = min(actual / expected, 1.0)
    filled = round(ratio * width)
    bar_color = color or (C.GREEN if ratio >= 1.0 else C.YELLOW)
    return f"{bar_color}{'█' * filled}{C.RESET}{C.DIM}{'░' * (width - filled)}{C.RESET}"


def _status_str(actual: int, expected: int) -> str:
    if actual >= expected:
        return f"{C.GREEN}DONE{C.RESET}"
    rem = expected - actual
    if actual == 0:
        return f"{C.RED}{rem} remaining{C.RESET}"
    return f"{C.YELLOW}{rem} remaining{C.RESET}"


def _count_missing(cur, from_clause: str, join_clause: str, where_null: str) -> int:
    """Count distinct run_ids in `results` that have no match in another table."""
    cur.execute(
        f"SELECT COUNT(DISTINCT r.run_id) FROM results r "
        f"LEFT JOIN {from_clause} ON {join_clause} "
        f"WHERE {where_null} IS NULL"
    )
    return cur.fetchone()[0]


# -- Sections -------------------------------------------------------------------

def section_db_info(conn: sqlite3.Connection):
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
    header("DISTINCT VALUES")
    cur = conn.cursor()
    cols = [
        "neural_dataset", "analysis", "compare_method", "region",
        "cfg_id", "seed", "pca_labels_folder", "model_name",
    ]
    for col in cols:
        cur.execute(f"SELECT DISTINCT {col} FROM results ORDER BY {col}")
        vals = [str(r[0]) for r in cur.fetchall()]
        print(f"  {col:25s} {', '.join(vals)}")


def _show_region_group(
    conn: sqlite3.Connection, cur, ds: str, analysis: str, cm: str,
    clause: str, n_subj: int, group_name: str | None, regions: list[str],
    show_reconstruction: bool,
):
    n_reg = len(regions)
    expected_per_config = n_subj * n_reg * N_SEEDS
    acolor = ANALYSIS_COLORS.get(analysis, "")

    label = f"{analysis} ({cm})"
    if group_name:
        label += f" -- {group_name}"
    subheader(label)

    region_list = ", ".join(f"'{r}'" for r in regions)
    region_filter = f"AND region IN ({region_list})"
    base_where = (
        f"neural_dataset = '{ds}' AND analysis = '{analysis}' "
        f"AND reconstruct_from_pcs = 0 {region_filter} {clause}"
    )
    tuple_expr = "subject_idx || '|' || region || '|' || seed"

    print(f"    regions ({n_reg}):  {', '.join(regions)}")
    print(
        f"    expected per config:  "
        f"{n_subj} subj x {n_reg} reg x {N_SEEDS} seeds = {expected_per_config}"
    )

    # Completeness per (pca_folder, cfg_id), excluding untrained (epoch=0)
    df = pd.read_sql(
        f"SELECT pca_labels_folder AS pca_folder, cfg_id, "
        f"COUNT(DISTINCT {tuple_expr}) AS actual "
        f"FROM results WHERE {base_where} AND epoch != 0 "
        f"GROUP BY pca_labels_folder, cfg_id "
        f"ORDER BY pca_labels_folder, cfg_id",
        conn,
    )

    if df.empty:
        print("    (no runs found)")
        return

    df["expected"] = expected_per_config

    print()
    for _, row in df.iterrows():
        pf, ci, act, exp = row["pca_folder"], int(row["cfg_id"]), int(row["actual"]), int(row["expected"])
        bar = progress_bar(act, exp, acolor)
        status = _status_str(act, exp)
        print(f"    {pf:>22s}  {ci:>5d}   {act:>3d}/{exp:<3d}  {bar}  {status}")

    total_actual = int(df["actual"].sum())
    total_expected = int(df["expected"].sum())
    total_pct = round(total_actual / total_expected * 100) if total_expected else 0
    n_done = int((df["actual"] >= df["expected"]).sum())
    print(
        f"\n    Total: {total_actual}/{total_expected} ({total_pct}%) "
        f"-- {n_done}/{len(df)} configs complete"
    )

    # Reconstruction info
    if show_reconstruction:
        cur.execute(
            f"SELECT COUNT(DISTINCT run_id), COUNT(DISTINCT pca_k) FROM results "
            f"WHERE neural_dataset = '{ds}' AND analysis = '{analysis}' "
            f"AND reconstruct_from_pcs = 1 {clause}"
        )
        n_recon, n_ks = cur.fetchone()
        if n_recon > 0:
            print(f"    + {n_recon} reconstruction runs ({n_ks} pca_k values)")

    # Baseline check (untrained: imagenet1k/1000 at epoch 0)
    expected_baseline = n_subj * n_reg * N_SEEDS
    cur.execute(
        f"SELECT COUNT(DISTINCT {tuple_expr}) FROM results "
        f"WHERE {base_where} AND pca_labels_folder = 'imagenet1k' "
        f"AND cfg_id = 1000 AND epoch = 0"
    )
    baseline_actual = cur.fetchone()[0]
    bar = progress_bar(baseline_actual, expected_baseline, acolor)
    status = _status_str(baseline_actual, expected_baseline)
    print(
        f"    {'Baseline (epoch 0)':>22s}         "
        f"{baseline_actual:>3d}/{expected_baseline:<3d}  {bar}  {status}"
    )

    # Missing diagnostics
    if total_actual < total_expected:
        cur.execute(
            f"SELECT seed, COUNT(DISTINCT subject_idx || '|' || region) AS n "
            f"FROM results WHERE {base_where} GROUP BY seed ORDER BY seed"
        )
        seed_coverage = cur.fetchall()
        full_per_seed = n_subj * n_reg
        seed_strs = []
        for seed, n in seed_coverage:
            pct = round(n / full_per_seed * 100)
            color = C.GREEN if pct == 100 else C.YELLOW if pct > 0 else C.RED
            seed_strs.append(f"{color}seed {seed}: {n}/{full_per_seed} ({pct}%){C.RESET}")
        for s in sorted(set(range(1, N_SEEDS + 1)) - {s for s, _ in seed_coverage}):
            seed_strs.append(f"{C.RED}seed {s}: 0/{full_per_seed} (0%){C.RESET}")
        print(f"    Seed coverage: {' | '.join(seed_strs)}")

        cur.execute(
            f"SELECT DISTINCT region FROM results "
            f"WHERE {base_where} ORDER BY region"
        )
        missing_regions = set(regions) - {r[0] for r in cur.fetchall()}
        if missing_regions:
            print(f"    Missing regions: {', '.join(sorted(missing_regions))}")


def section_completeness(conn: sqlite3.Connection, where: str):
    header("COMPLETENESS")

    clause = f"AND {where}" if where else ""
    cur = conn.cursor()

    cur.execute(
        f"SELECT DISTINCT neural_dataset, analysis, compare_method "
        f"FROM results WHERE 1=1 {clause} ORDER BY neural_dataset, analysis"
    )
    combos = cur.fetchall()

    for ds, ds_combos in groupby(combos, key=lambda x: x[0]):
        dataset_header(ds)
        for _, analysis, cm in ds_combos:
            anatomy = EXPECTED_ANATOMY.get((ds, analysis))
            if not anatomy:
                subheader(f"{analysis} ({cm})  [no expected anatomy defined]")
                continue

            n_subj = anatomy["n_subjects"]
            for group in anatomy["groups"]:
                _show_region_group(
                    conn, cur, ds, analysis, cm, clause,
                    n_subj, group["name"], group["regions"],
                    group["has_reconstruction"],
                )


def section_health(conn: sqlite3.Connection):
    header("HEALTH CHECKS")
    cur = conn.cursor()

    orphaned = _count_missing(
        cur, "run_configs rc", "r.run_id = rc.run_id", "rc.run_id"
    )
    status = f"{C.GREEN}OK{C.RESET}" if orphaned == 0 else f"{C.RED}WARN: {orphaned} orphaned{C.RESET}"
    print(f"  run_configs coverage:       {status}")

    total = pd.read_sql("SELECT COUNT(DISTINCT run_id) AS n FROM results", conn).iloc[0, 0]

    no_boot = _count_missing(
        cur,
        "bootstrap_distributions bd",
        "r.run_id = bd.run_id AND r.compare_method = bd.compare_method",
        "bd.run_id",
    )
    if no_boot == 0:
        print(f"  bootstrap_distributions:    {C.GREEN}OK{C.RESET} ({total}/{total})")
    else:
        print(f"  bootstrap_distributions:    {C.YELLOW}{total - no_boot}/{total}{C.RESET} have bootstrap")

    no_ls = _count_missing(
        cur,
        "(SELECT DISTINCT run_id FROM layer_selection_scores) ls",
        "r.run_id = ls.run_id",
        "ls.run_id",
    )
    if no_ls == 0:
        print(f"  layer_selection_scores:     {C.GREEN}OK{C.RESET} ({total}/{total})")
    else:
        print(f"  layer_selection_scores:     {C.YELLOW}{total - no_ls}/{total}{C.RESET} have layer selection")
        df_ls = pd.read_sql(
            "SELECT r.neural_dataset, r.reconstruct_from_pcs AS recon, "
            "COUNT(DISTINCT r.run_id) AS missing FROM results r "
            "LEFT JOIN (SELECT DISTINCT run_id FROM layer_selection_scores) ls "
            "ON r.run_id = ls.run_id WHERE ls.run_id IS NULL "
            "GROUP BY r.neural_dataset, r.reconstruct_from_pcs "
            "ORDER BY r.neural_dataset, r.reconstruct_from_pcs",
            conn,
        )
        recon_missing = df_ls[df_ls["recon"] == 1]["missing"].sum()
        baseline_missing = df_ls[df_ls["recon"] == 0]["missing"].sum()
        print(f"    reconstruction sweep (expected):  {recon_missing}")
        if baseline_missing > 0:
            datasets = ", ".join(df_ls[df_ls["recon"] == 0]["neural_dataset"].tolist())
            print(f"    baseline runs (reuses parent):    {baseline_missing}  ({datasets})")

    cur.execute("SELECT COUNT(*) FROM results WHERE score IS NULL")
    null_scores = cur.fetchone()[0]
    status = f"{C.GREEN}OK{C.RESET}" if null_scores == 0 else f"{C.RED}WARN: {null_scores} rows{C.RESET}"
    print(f"  NULL scores:                {status}")


def section_recent(conn: sqlite3.Connection, n: int):
    header(f"RECENT RUNS  (last {n})")
    df = pd.read_sql(
        f"SELECT rc.created_at, r.neural_dataset, r.analysis, "
        f"r.pca_labels_folder, r.cfg_id, r.seed, r.region, r.subject_idx "
        f"FROM run_configs rc JOIN results r ON rc.run_id = r.run_id "
        f"ORDER BY rc.created_at DESC LIMIT {n}",
        conn,
    )
    print_df(df)


def run_custom_query(conn: sqlite3.Connection, query: str):
    header("CUSTOM QUERY")
    print(f"  {query}\n")
    df = pd.read_sql(query, conn)
    with pd.option_context("display.max_rows", 60, "display.width", 160):
        print_df(df)
    print(f"\n  ({len(df)} rows returned)")


# -- Main -----------------------------------------------------------------------

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
