import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from visreps.dataloaders.obj_cls import get_obj_cls_loader, DS_MEAN, DS_STD


def verify_pca_labels(dataset, n_classes: int) -> None:
    """Verify that PCA labels match those in the CSV file."""
    csv_path = f"datasets/obj_cls/tiny-imagenet/pca_labels/n_classes_{n_classes}.csv"
    if not os.path.exists(csv_path):
        print(f"! Warning: PCA label file {csv_path} not found")
        return
    
    # Read top 10 rows from CSV
    df = pd.read_csv(csv_path, nrows=10)
    print(f"\nVerifying first {len(df)} images from PCA labels file:")
    print(df)
    
    # Get the base dataset
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    # Create a mapping of image names to indices in the dataset
    image_to_idx = {os.path.basename(sample[2]): i 
                    for i, sample in enumerate(base_dataset.samples)}
    
    # Verify each image from CSV
    for _, row in df.iterrows():
        img_name = row['image']
        expected_label = row['pca_label']
        
        if img_name not in image_to_idx:
            print(f"! Warning: Image {img_name} not found in dataset")
            continue
            
        idx = image_to_idx[img_name]
        actual_image, actual_label = dataset[idx]
        
        if actual_label != expected_label:
            raise AssertionError(
                f"Label mismatch for {img_name}: "
                f"expected {expected_label}, got {actual_label}"
            )
    
    print("✓ PCA labels verified successfully for sampled images")


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization using Tiny ImageNet statistics.
    """
    img = img.clone()
    mean = DS_MEAN["tiny-imagenet"]
    std = DS_STD["tiny-imagenet"]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor to a PIL Image."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.clone()
    mean = DS_MEAN["tiny-imagenet"]
    std = DS_STD["tiny-imagenet"]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)
    img_uint8 = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(img_uint8)


def visualize_batch(images: torch.Tensor, labels: torch.Tensor, save_path: str) -> None:
    """
    Create and save a grid visualization of a batch of images.
    """
    grid = make_grid(images, nrow=8, padding=2)
    grid = denormalize_image(grid)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Batch Visualization")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def test_tiny_imagenet_loader() -> None:
    """
    Test Tiny ImageNet dataloader functionality for both standard and PCA label cases.
    Tests data splits, batch shapes, dtypes, value ranges, and augmentations.
    """
    print("\nTesting Tiny ImageNet dataloader...")

    base_cfg = {
        "dataset": "tiny-imagenet",
        "dataset_path": "datasets/obj_cls/tiny-imagenet",
        "batchsize": 32,
        "num_workers": 4,
        "data_augment": True,
    }

    # Test cases: with and without PCA labels
    test_configs = [
        {**base_cfg, "pca_labels": False},
        {**base_cfg, "pca_labels": True, "pca_n_classes": 2},
    ]

    for cfg in test_configs:
        pca_mode = "PCA" if cfg["pca_labels"] else "Standard"
        print(f"\n=== Testing {pca_mode} Label Mode ===")

        try:
            datasets, loaders = get_obj_cls_loader(cfg)
            print(f"✓ Created datasets and dataloaders in {pca_mode} mode")

            # Verify splits
            assert "train" in loaders and "test" in loaders, "Missing data splits"
            print(f"✓ Found train samples: {len(datasets['train'])}, test samples: {len(datasets['test'])}")

            # Verify PCA labels if applicable
            if cfg["pca_labels"]:
                verify_pca_labels(datasets["train"], cfg["pca_n_classes"])

            # Check label consistency using batches
            train_labels = set()
            test_labels = set()
            
            # Sample a few batches for checking labels
            num_batches_to_check = 10
            print(f"\nChecking {num_batches_to_check} batches for label consistency...")
            
            for i, (_, labels) in enumerate(loaders["train"]):
                if i >= num_batches_to_check:
                    break
                train_labels.update(labels.tolist())
                
            for i, (_, labels) in enumerate(loaders["test"]):
                if i >= num_batches_to_check:
                    break
                test_labels.update(labels.tolist())

            # Expected label range based on mode
            if cfg["pca_labels"]:
                max_label = cfg["pca_n_classes"]
                assert max(train_labels) < max_label, f"PCA labels exceed {max_label}"
                assert max(test_labels) < max_label, f"PCA labels exceed {max_label}"
                print(f"✓ PCA labels in range [0, {max_label}]")
                print(f"✓ Found {len(train_labels)} unique PCA classes in sampled batches")
            else:
                assert max(train_labels) < 200, "Standard labels exceed class count"
                assert test_labels.issubset(set(range(200))), "Invalid standard labels found"
                print("✓ Standard labels in range [0, 199]")
                print(f"✓ Found {len(train_labels)} unique classes in sampled batches")

            # Test batch properties
            images, labels = next(iter(loaders["train"]))
            
            # Shape checks
            assert images.shape == (cfg["batchsize"], 3, 64, 64), f"Wrong image shape: {images.shape}"
            assert labels.shape == (cfg["batchsize"],), f"Wrong label shape: {labels.shape}"
            print(f"✓ Batch shapes correct: images {images.shape}, labels {labels.shape}")

            # Type checks
            assert images.dtype == torch.float32, "Images should be float32"
            assert labels.dtype == torch.long, "Labels should be long"
            print("✓ Data types correct")

            # Print sample batch info
            print("\nSample batch labels:", labels[:5].tolist())
            print("Sample image value range:", f"[{images.min():.3f}, {images.max():.3f}]")

            # Test augmentation consistency
            if cfg["data_augment"]:
                base_dataset = datasets["train"].dataset if cfg["pca_labels"] else datasets["train"]
                aug_transforms = [t for t in base_dataset.transform.transforms
                                if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation))]
                
                if aug_transforms:
                    print("✓ Data augmentation transforms present:", [type(t).__name__ for t in aug_transforms])
                else:
                    print("! No augmentation transforms found")

            print(f"\n{pca_mode} mode tests passed successfully!")

        except Exception as e:
            print(f"❌ {pca_mode} mode test failed: {str(e)}")
            raise

    print("\nAll tests completed! 🎉")


if __name__ == "__main__":
    test_tiny_imagenet_loader()