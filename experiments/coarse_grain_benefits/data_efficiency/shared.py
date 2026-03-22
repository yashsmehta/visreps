"""Shared constants and utilities for the data efficiency experiment pipeline."""

import os

import pandas as pd

SEED = 1
SEED_LETTER = "a"
NUM_EPOCHS = 200
DEFAULT_PCA_LABELS = "clip"
DATASETS = ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]
EPOCHS = [100, 200]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_conditions(pca_labels):
    """Build conditions dict using the specified PCA labels folder."""
    folder = f"pca_labels_{pca_labels}"
    return {
        8:    {"pca_labels": True, "pca_n_classes": 8,  "pca_labels_folder": folder},
        16:   {"pca_labels": True, "pca_n_classes": 16, "pca_labels_folder": folder},
        32:   {"pca_labels": True, "pca_n_classes": 32, "pca_labels_folder": folder},
        64:   {"pca_labels": True, "pca_n_classes": 64, "pca_labels_folder": folder},
        1000: {"pca_labels": False, "pca_n_classes": 1000},
    }


def get_checkpoint_dir(dataset, pca_labels):
    """Build checkpoint directory name (relative to model_checkpoints/)."""
    if pca_labels == DEFAULT_PCA_LABELS:
        return f"data_efficiency_{dataset}"
    return f"data_efficiency_{pca_labels}_{dataset}"


def get_csv_path(pca_labels):
    """Return CSV path, including PCA label source if non-default."""
    if pca_labels == DEFAULT_PCA_LABELS:
        return os.path.join(SCRIPT_DIR, "data_efficiency_results.csv")
    return os.path.join(SCRIPT_DIR, f"data_efficiency_{pca_labels}_results.csv")


def save_results(rows, csv_path):
    """Append result rows to the combined CSV, deduplicating by key columns."""
    keys = ["dataset", "condition", "epoch", "benchmark", "subject_idx"]
    new_df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(keys).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
