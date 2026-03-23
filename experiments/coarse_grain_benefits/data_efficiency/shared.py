"""Shared constants and utilities for the data efficiency experiment pipeline."""

import os

import pandas as pd

SEED = 1
SEED_LETTER = "a"
NUM_EPOCHS = 100
DEFAULT_PCA_LABELS = "clip"
DATASETS = ["imagenet-mini-10", "imagenet-mini-100"]
EPOCHS = [50, 100]
CHECKPOINT_BASE = "/data/ymehta3/data_efficiency"

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


def get_checkpoint_dir(dataset, pca_labels, condition_id=None):
    """Build checkpoint directory name (relative to CHECKPOINT_BASE).

    Coarse conditions (8-64) and fine-grained (1000) live in separate folders:
      customcnn_clip_imagenet-mini-{N}  (coarse, CLIP PCA)
      customcnn_imagenet-mini-{N}       (1000-class)
    """
    if condition_id == 1000:
        return f"customcnn_{dataset}"
    return f"customcnn_{pca_labels}_{dataset}"


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
