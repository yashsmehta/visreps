import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from visreps.dataloaders.obj_cls import get_obj_cls_loader, DS_MEAN, DS_STD


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
    """
    Convert a normalized tensor to a PIL Image for augmentation.
    
    Accepts a tensor of shape (C, H, W) or a batch (N, C, H, W). If a batch is provided,
    only the first image is converted.
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]  # Use the first image from the batch
    img_denorm = denormalize_image(img_tensor)
    # Change from (C, H, W) to (H, W, C) and scale to [0, 255]
    img_uint8 = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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
    Test Tiny ImageNet dataloader functionality including:
      - Data splits and label consistency
      - Batch shape, dtype, and value range checks
      - Sample visualization
      - Data augmentation checks
      - PCA label integration
    """
    print("Testing Tiny ImageNet dataloader...")

    # Configuration
    cfg = {
        "dataset": "tiny-imagenet",
        "dataset_path": "datasets/obj_cls/tiny-imagenet",
        "batchsize": 32,
        "num_workers": 4,
        "data_augment": True,
        "n_bits": 2,  # For PCA labels testing
    }

    try:
        # Test with original labels
        datasets, loaders = get_obj_cls_loader(cfg, pca_labels=False)
        print("✓ Successfully created datasets and dataloaders with original labels")

        # Basic checks for data splits
        if "train" not in loaders or "test" not in loaders:
            raise AssertionError("Missing train or test splits")
        print("✓ Both train and test splits present")

        # Verify label consistency using the samples attribute
        train_labels = {label for _, label, _ in datasets["train"].samples}
        test_labels = {label for _, label, _ in datasets["test"].samples}
        all_labels = train_labels.union(test_labels)
        if not all(isinstance(label, int) for label in all_labels):
            raise AssertionError("All labels should be integers")
        if not (min(all_labels) >= 0 and max(all_labels) < 200):
            raise AssertionError("Labels should be in range [0, 199]")
        print(f"✓ Found {len(all_labels)} unique labels")

        # Test batch loading for train and test splits
        train_batch = next(iter(loaders["train"]))
        test_batch = next(iter(loaders["test"]))
        if len(train_batch) != 2:
            raise AssertionError("Batch should contain (images, labels)")
        images, labels = train_batch
        expected_shape = (cfg["batchsize"], 3, 64, 64)
        if images.shape != expected_shape:
            raise AssertionError(f"Unexpected image shape: {images.shape}")
        if labels.shape != (cfg["batchsize"],):
            raise AssertionError(f"Unexpected labels shape: {labels.shape}")
        print("✓ Batch shapes are correct")

        # Check data types and label value ranges
        if images.dtype != torch.float32:
            raise AssertionError("Images should be float32")
        if labels.dtype != torch.long:
            raise AssertionError("Labels should be long")
        if not all(0 <= label.item() < 200 for label in labels):
            raise AssertionError("Labels should be in range [0, 199]")
        print("✓ Data types and value ranges are correct")

        # Print sample labels
        print("\nSample batch info:")
        for i in range(min(5, len(labels))):
            print(f"  Label: {labels[i].item()}")

        # Visualize samples
        os.makedirs("tests/outputs", exist_ok=True)
        visualize_batch(images, labels, save_path="tests/outputs/tiny_imagenet_samples.png")
        print("✓ Saved sample visualization to tests/outputs/tiny_imagenet_samples.png")

        # Test data augmentation
        test_images, test_labels = test_batch
        test_img = test_images[0]  # Use first image from test batch
        pil_img = tensor_to_pil(test_img)
        augmented_tensors = []

        # Extract augmentation transforms (only horizontal flip and rotation)
        aug_transforms = [
            t for t in datasets["train"].dataset.transform.transforms
            if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation))
        ]
        aug_compose = transforms.Compose(aug_transforms)

        for _ in range(5):
            aug_pil = aug_compose(pil_img)
            aug_tensor = F.normalize(
                F.to_tensor(aug_pil), DS_MEAN["tiny-imagenet"], DS_STD["tiny-imagenet"]
            )
            augmented_tensors.append(aug_tensor)

        # Visualize augmented samples
        plt.figure(figsize=(15, 3))
        aug_grid = make_grid(augmented_tensors, nrow=5, padding=2)
        plt.imshow(denormalize_image(aug_grid).permute(1, 2, 0).cpu().numpy())
        plt.title("Augmentation Examples (Same Image)", fontsize=12)
        plt.axis("off")
        plt.savefig("tests/outputs/tiny_imagenet_augmentation_samples.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("✓ Saved augmentation visualization to tests/outputs/tiny_imagenet_augmentation_samples.png")

        # Test with PCA labels
        pca_datasets, pca_loaders = get_obj_cls_loader(cfg, pca_labels=True)
        print("\nTesting PCA label integration...")
        pca_batch = next(iter(pca_loaders["train"]))
        if len(pca_batch) != 2:
            raise AssertionError("PCA batch should contain (images, labels)")
        images, labels = pca_batch

        # Verify PCA label ranges (should be 0 to 2^n_bits - 1)
        n_classes = 2 ** cfg["n_bits"]
        if not all(0 <= label.item() < n_classes for label in labels):
            raise AssertionError(f"PCA labels should be in range [0, {n_classes - 1}]")
        print(f"✓ PCA labels are correctly bounded [0, {n_classes - 1}]")

        print("\nAll tests passed successfully! 🎉")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_tiny_imagenet_loader()