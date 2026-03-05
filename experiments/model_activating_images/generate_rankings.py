"""Rank ImageNet images by output logit for each class of a trained model.

Loads a checkpoint, runs inference on ImageNet (full or subsampled), and saves
the top-N images per output class (ranked by logit) to a CSV.

Usage:
    python experiments/model_activating_images/generate_rankings.py
    python experiments/model_activating_images/generate_rankings.py --checkpoint_path /data/ymehta3/alexnet_pca/cfg4a/checkpoint_epoch_20.pth --n_subsample 50000
"""

import os
import sys
import argparse
import random

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Backward-compat shim: older checkpoints may reference 'visreps.models.custom_cnn'
from visreps.models import custom_model
sys.modules['visreps.models.custom_cnn'] = custom_model

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from visreps.dataloaders.obj_cls import ImageNetDataset, get_transform, create_dataloader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_imagenet_class_mapping(imagenet_data_dir):
    """Load ImageNet class ID to human-readable name from map_clsloc.txt."""
    mapping_file = os.path.join(imagenet_data_dir, "map_clsloc.txt")
    class_mapping = {}
    if not os.path.exists(mapping_file):
        return class_mapping
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    class_mapping[parts[0]] = parts[1]
    return class_mapping


def main():
    parser = argparse.ArgumentParser(description="Rank ImageNet images by model output logit")
    parser.add_argument('--checkpoint_path', type=str,
                        default='/data/ymehta3/alexnet_pca/cfg2a/checkpoint_epoch_20.pth')
    parser.add_argument('--n_top', type=int, default=100,
                        help="Number of top images to save per class")
    parser.add_argument('--n_subsample', type=int, default=None,
                        help="Subsample N images (default: use all)")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_csv', type=str, default=None,
                        help="Output CSV path (default: rankings.csv in script dir)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, weights_only=False, map_location=device)
    model = ckpt['model'].to(device)
    model.eval()

    n_classes = model.classifier[-1].out_features
    print(f"Model has {n_classes} output classes")

    # --- Load ImageNet ---
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR")
    if not imagenet_dir:
        raise EnvironmentError("IMAGENET_DATA_DIR not set. Source .env first.")

    class_mapping = load_imagenet_class_mapping(imagenet_dir)

    transform = get_transform("imgnet", data_augment=False)
    dataset = ImageNetDataset(imagenet_dir, split="all", transform=transform)
    print(f"Full dataset: {len(dataset)} images")

    # Optional subsampling
    if args.n_subsample and args.n_subsample < len(dataset):
        rng = random.Random(42)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        indices = indices[:args.n_subsample]
        loader_dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subsampled to {len(loader_dataset)} images")
    else:
        indices = None
        loader_dataset = dataset

    loader = create_dataloader(
        loader_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )

    # --- Forward pass ---
    print("Running inference...")
    all_logits = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Inference"):
            logits = model(images.to(device, non_blocking=True))
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    print(f"Collected logits for {all_logits.shape[0]} images")

    # --- Rank per class ---
    rows = []
    for class_idx in range(n_classes):
        logits_c = all_logits[:, class_idx].numpy()
        top_local = np.argsort(logits_c)[::-1][:args.n_top]

        for rank, local_idx in enumerate(top_local):
            global_idx = indices[local_idx] if indices else local_idx
            img_file = dataset.samples[global_idx][2]
            wnid = img_file.split('_')[0]
            rows.append({
                'class_idx': class_idx,
                'rank': rank + 1,
                'logit': float(logits_c[local_idx]),
                'image_file': img_file,
                'imagenet_class_id': wnid,
                'imagenet_class_name': class_mapping.get(wnid, "unknown"),
            })

    df = pd.DataFrame(rows)

    # --- Save ---
    output_csv = args.output_csv or os.path.join(SCRIPT_DIR, "rankings.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to {output_csv}")

    # --- Summary ---
    for c in range(n_classes):
        df_c = df[df['class_idx'] == c]
        top5_classes = df_c.head(20)['imagenet_class_name'].value_counts().head(5)
        print(f"\nClass {c} — top ImageNet classes among top-20:")
        for name, count in top5_classes.items():
            print(f"  {name}: {count}")


if __name__ == '__main__':
    main()
