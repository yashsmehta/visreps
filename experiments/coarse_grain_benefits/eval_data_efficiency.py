"""
Evaluate data-efficiency trained models on THINGS behavioral alignment.
Compares 8-class (CLIP) vs 1000-class models trained on imagenet-mini-50.

Optionally compares against full-ImageNet baselines from results.db.

Usage (from project root):
    python experiments/coarse_grain_benefits/eval_data_efficiency.py
    python experiments/coarse_grain_benefits/eval_data_efficiency.py --epoch 200
    python experiments/coarse_grain_benefits/eval_data_efficiency.py --skip_baselines
"""

import os
import sys
import sqlite3
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
import visreps.evals as evals
from visreps.utils import load_config, validate_config

CHECKPOINT_DIR = "model_checkpoints/data_efficiency"
SEED = 1

CONDITIONS = [
    {"label": "8-class (CLIP, mini-50)", "cfg_id": 8, "pca_labels": True,
     "pca_n_classes": 8, "pca_labels_folder": "pca_labels_clip"},
    {"label": "1000-class (mini-50)", "cfg_id": 1000, "pca_labels": False,
     "pca_n_classes": 1000},
]


def eval_condition(condition, epoch):
    """Evaluate a single condition on THINGS."""
    label = condition["label"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {label} (epoch {epoch})")
    print(f"{'='*60}\n")

    overrides = [
        f"seed={SEED}",
        f"cfg_id={condition['cfg_id']}",
        f"checkpoint_dir={CHECKPOINT_DIR}",
        f"checkpoint_model=checkpoint_epoch_{epoch}.pth",
        f"pca_labels={condition['pca_labels']}",
        f"pca_n_classes={condition['pca_n_classes']}",
        "neural_dataset=things-behavior",
        "analysis=rsa",
        "compare_method=spearman",
        "bootstrap=true",
        "load_model_from=checkpoint",
        "batchsize=256",
    ]
    if "pca_labels_folder" in condition:
        overrides.append(f"pca_labels_folder={condition['pca_labels_folder']}")

    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    evals.eval(cfg)
    print(f"\n{label} evaluation complete.")


def print_comparison(epoch, skip_baselines=False):
    """Print results comparison table."""
    db_path = os.path.join(PROJECT_ROOT, "results.db")
    if not os.path.exists(db_path):
        print("No results.db found.")
        return

    conn = sqlite3.connect(db_path)
    try:
        # Get mini-50 results
        mini_df = pd.read_sql("""
            SELECT cfg_id, pca_n_classes, score, ci_low, ci_high, layer
            FROM results
            WHERE neural_dataset = 'things-behavior'
              AND compare_method = 'spearman'
              AND seed = 1
              AND checkpoint_dir = ?
              AND epoch = ?
            ORDER BY cfg_id
        """, conn, params=(CHECKPOINT_DIR, epoch))

        if len(mini_df) == 0:
            print("No mini-50 results found in database yet.")
            return

        print(f"\n{'='*70}")
        print(f"THINGS Behavioral Alignment — Data Efficiency Results (epoch {epoch})")
        print(f"{'='*70}")
        print(f"{'Condition':<30} {'Score':>8} {'CI':>20} {'Layer':>8}")
        print(f"{'-'*70}")

        for _, row in mini_df.iterrows():
            label = f"{int(row['cfg_id'])}-class (mini-50)"
            ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if row['ci_low'] else ""
            print(f"{label:<30} {row['score']:>8.4f} {ci:>20} {row['layer']:>8}")

        # Show baselines if available
        if not skip_baselines:
            baselines_df = pd.read_sql("""
                SELECT cfg_id, score, ci_low, ci_high, layer
                FROM results
                WHERE neural_dataset = 'things-behavior'
                  AND compare_method = 'spearman'
                  AND seed = 1
                  AND cfg_id IN (8, 1000)
                  AND checkpoint_dir != ?
                ORDER BY cfg_id
            """, conn, params=(CHECKPOINT_DIR,))

            if len(baselines_df) > 0:
                print(f"{'-'*70}")
                print("Full ImageNet baselines:")
                for _, row in baselines_df.iterrows():
                    label = f"{int(row['cfg_id'])}-class (full ImageNet)"
                    ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if row['ci_low'] else ""
                    print(f"{label:<30} {row['score']:>8.4f} {ci:>20} {row['layer']:>8}")

        print(f"{'='*70}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate data efficiency models on THINGS")
    parser.add_argument("--epoch", type=int, default=200, help="Checkpoint epoch to evaluate")
    parser.add_argument("--conditions", type=int, nargs="+", default=[8, 1000],
                        choices=[8, 1000], help="Which conditions to evaluate")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--print_only", action="store_true", help="Only print results, don't run eval")
    args = parser.parse_args()

    if not args.print_only:
        for cond in CONDITIONS:
            if cond["cfg_id"] in args.conditions:
                eval_condition(cond, args.epoch)

    print_comparison(args.epoch, skip_baselines=args.skip_baselines)


if __name__ == "__main__":
    main()
