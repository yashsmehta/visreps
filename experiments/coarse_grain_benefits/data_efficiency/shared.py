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

# Maps CLI architecture name → checkpoint directory prefix
ARCHITECTURES = {
    "customcnn": "customcnn",
    "resnet50": "resnet50",
    "vit_base": "vit_base",
    "convnext_base": "convnext_base",
}

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


def get_checkpoint_dir(dataset, pca_labels, condition_id=None, arch="customcnn"):
    """Build checkpoint directory name (relative to CHECKPOINT_BASE).

    Coarse conditions (8-64) and fine-grained (1000) live in separate folders:
      {arch_prefix}_{pca_labels}_{dataset}  (coarse, e.g. resnet50_clip_imagenet-mini-10)
      {arch_prefix}_{dataset}               (1000-class, e.g. resnet50_imagenet-mini-10)
    """
    prefix = ARCHITECTURES[arch]
    if condition_id == 1000:
        return f"{prefix}_{dataset}"
    return f"{prefix}_{pca_labels}_{dataset}"


def get_csv_path(pca_labels, arch="customcnn"):
    """Return CSV path, including architecture and PCA label source if non-default."""
    parts = ["data_efficiency"]
    if arch != "customcnn":
        parts.append(arch)
    if pca_labels != DEFAULT_PCA_LABELS:
        parts.append(pca_labels)
    parts.append("results")
    return os.path.join(SCRIPT_DIR, f"{'_'.join(parts)}.csv")


def save_results(rows, csv_path):
    """Append result rows to the combined CSV, deduplicating by key columns."""
    keys = ["dataset", "condition", "epoch", "benchmark", "region", "subject_idx"]
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
