"""
Evaluate data-efficiency trained models on THINGS behavioral alignment.
Compares 8-class (CLIP) vs 1000-class models trained on imagenet-mini subsets.
Results are saved directly to data_efficiency/data_efficiency.csv (no results.db).

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --dataset imagenet-mini-50
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --dataset imagenet-mini-10 --epoch 200
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --dataset imagenet-mini-50 --print_only
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
import visreps.evals as evals
from visreps.utils import load_config, validate_config

SEED = 1
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_efficiency.csv")
CSV_COLUMNS = ["dataset", "condition", "epoch", "score", "ci_low", "ci_high", "layer"]

CONDITIONS = [
    {"label": "8-class (CLIP)", "cfg_id": 8, "pca_labels": True,
     "pca_n_classes": 8, "pca_labels_folder": "pca_labels_clip"},
    {"label": "1000-class", "cfg_id": 1000, "pca_labels": False,
     "pca_n_classes": 1000},
]


def get_checkpoint_dir(dataset):
    return f"model_checkpoints/data_efficiency_{dataset}"


def save_to_csv(dataset, condition_cfg_id, epoch, result_df):
    """Upsert a result row into the CSV, keyed on (dataset, condition, epoch)."""
    row = result_df.iloc[0]
    new_row = pd.DataFrame([{
        "dataset": dataset,
        "condition": condition_cfg_id,
        "epoch": epoch,
        "score": round(row["score"], 4),
        "ci_low": round(row["ci_low"], 4),
        "ci_high": round(row["ci_high"], 4),
        "layer": row["layer"],
    }])

    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        # Drop any existing row with the same key
        mask = (
            (existing["dataset"] == dataset) &
            (existing["condition"] == condition_cfg_id) &
            (existing["epoch"] == epoch)
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined = combined.sort_values(["dataset", "condition", "epoch"]).reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"  Saved to {CSV_PATH}")


def eval_condition(condition, epoch, dataset):
    """Evaluate a single condition on THINGS and save to CSV."""
    checkpoint_dir = get_checkpoint_dir(dataset)
    label = f"{condition['label']} ({dataset})"
    print(f"\n{'='*60}")
    print(f"Evaluating: {label} (epoch {epoch})")
    print(f"{'='*60}\n")

    overrides = [
        f"seed={SEED}",
        f"cfg_id={condition['cfg_id']}",
        f"checkpoint_dir={checkpoint_dir}",
        f"checkpoint_model=checkpoint_epoch_{epoch}.pth",
        f"pca_labels={condition['pca_labels']}",
        f"pca_n_classes={condition['pca_n_classes']}",
        "neural_dataset=things-behavior",
        "analysis=rsa",
        "compare_method=spearman",
        "bootstrap=true",
        "load_model_from=checkpoint",
        "log_expdata=false",
        "batchsize=256",
    ]
    if "pca_labels_folder" in condition:
        overrides.append(f"pca_labels_folder={condition['pca_labels_folder']}")

    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)

    save_to_csv(dataset, condition["cfg_id"], epoch, result_df)
    print(f"\n{label} evaluation complete.")


def print_comparison(dataset, skip_baselines=False):
    """Print results comparison table from CSV."""
    if not os.path.exists(CSV_PATH):
        print("No data_efficiency.csv found.")
        return

    df = pd.read_csv(CSV_PATH)
    mini_df = df[df["dataset"] == dataset].sort_values(["condition", "epoch"])

    if len(mini_df) == 0:
        print(f"No {dataset} results found in CSV.")
        return

    print(f"\n{'='*70}")
    print(f"THINGS Alignment — {dataset}")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Epoch':>6} {'Score':>8} {'CI':>22} {'Layer':>10}")
    print(f"{'-'*70}")

    for _, row in mini_df.iterrows():
        label = f"{int(row['condition'])}-class"
        ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        print(f"{label:<25} {int(row['epoch']):>6} {row['score']:>8.4f} {ci:>22} {row['layer']:>10}")

    if not skip_baselines:
        baselines = df[df["dataset"] == "imagenet-full"].sort_values(["condition", "epoch"])
        if len(baselines) > 0 and dataset != "imagenet-full":
            print(f"{'-'*70}")
            print("Full ImageNet baselines:")
            for _, row in baselines.iterrows():
                label = f"{int(row['condition'])}-class (full)"
                ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                print(f"{label:<25} {int(row['epoch']):>6} {row['score']:>8.4f} {ci:>22} {row['layer']:>10}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate data efficiency models on THINGS")
    parser.add_argument("--dataset", type=str, default="imagenet-mini-50",
                        choices=["imagenet-mini-10", "imagenet-mini-50", "imagenet-mini-200"])
    parser.add_argument("--epoch", type=int, default=200, help="Checkpoint epoch to evaluate")
    parser.add_argument("--conditions", type=int, nargs="+", default=[8, 1000],
                        choices=[8, 1000], help="Which conditions to evaluate")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--print_only", action="store_true", help="Only print results, don't run eval")
    args = parser.parse_args()

    if not args.print_only:
        for cond in CONDITIONS:
            if cond["cfg_id"] in args.conditions:
                eval_condition(cond, args.epoch, args.dataset)

    print_comparison(args.dataset, skip_baselines=args.skip_baselines)


if __name__ == "__main__":
    main()
