import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

from visreps.dataloaders.obj_cls import get_obj_cls_loader, DS_MEAN, DS_STD

def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor for visualization using Tiny ImageNet stats"""
    img = img.clone()
    mean = DS_MEAN["tiny-imagenet"]
    std = DS_STD["tiny-imagenet"]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)

def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor to PIL Image for augmentation"""
    img_denorm = denormalize_image(img_tensor)
    img_uint8 = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

def visualize_batch(images: torch.Tensor, labels: torch.Tensor, save_path: str = None):
    """Visualize a batch of images in a clean 2x4 grid"""
    # Set style
    sns.set_style("white")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Denormalize images
    images = denormalize_image(images)
    
    # Create figure with subplots (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    
    # Plot each image
    for idx, (ax, img) in enumerate(zip(axes_flat, images[:8])):
        ax.imshow(img.permute(1, 2, 0).cpu())
        ax.axis('off')
    
    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()

def test_tiny_imagenet_loader():
    """Test Tiny ImageNet dataloader functionality"""
    print("Testing Tiny ImageNet dataloader...")
    
    # Configuration
    cfg = {
        "dataset": "tiny-imagenet",
        "dataset_path": "datasets/obj_cls/tiny-imagenet-200",
        "batchsize": 32,
        "num_workers": 4,
        "data_augment": True,
        "n_bits": 2  # For PCA labels testing
    }
    
    try:
        # Test with original labels
        datasets, loaders = get_obj_cls_loader(cfg, pca_labels=False)
        print("✓ Successfully created datasets and dataloaders with original labels")
        
        # Basic checks
        assert "train" in loaders and "test" in loaders, "Missing train or test splits"
        print("✓ Both train and test splits present")
        
        # Verify label consistency
        train_labels = set(label for _, label, _ in datasets["train"].samples)
        test_labels = set(label for _, label, _ in datasets["test"].samples)
        all_labels = train_labels.union(test_labels)
        assert all(isinstance(label, int) for label in all_labels), "All labels should be integers"
        assert max(all_labels) < 200 and min(all_labels) >= 0, "Labels should be in range [0, 199]"
        print(f"✓ Found {len(all_labels)} unique labels")
        
        # Test batch loading
        train_batch = next(iter(loaders["train"]))
        test_batch = next(iter(loaders["test"]))
        
        # Check shapes
        assert len(train_batch) == 3, "Batch should contain (images, labels, img_ids)"
        images, labels, img_ids = train_batch
        assert images.shape == (cfg["batchsize"], 3, 224, 224), f"Unexpected image shape: {images.shape}"
        assert labels.shape == (cfg["batchsize"],), f"Unexpected labels shape: {labels.shape}"
        print("✓ Batch shapes are correct")
        
        # Check value ranges
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.long, "Labels should be long"
        assert all(0 <= label.item() < 200 for label in labels), "Labels should be in range [0, 199]"
        print("✓ Data types are correct")
        
        # Print some sample labels and image IDs
        print("\nSample batch info:")
        for i in range(min(5, len(labels))):
            print(f"  Image: {img_ids[i]}, Label: {labels[i].item()}")
        
        # Visualize samples
        os.makedirs("tests/outputs", exist_ok=True)
        visualize_batch(images, labels, save_path="tests/outputs/tiny_imagenet_samples.png")
        print("✓ Saved sample visualization to tests/outputs/tiny_imagenet_samples.png")
        
        # Test data augmentation
        test_img = test_batch[0][0]  # First image from test batch
        pil_img = tensor_to_pil(test_img)
        train_augmented = []
        
        # Get just the augmentation transforms
        aug_transforms = [t for t in datasets["train"].dataset.transform.transforms 
                        if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation))]
        aug_compose = transforms.Compose(aug_transforms)
        
        # Apply augmentations and convert back to normalized tensor
        for _ in range(5):
            aug_pil = aug_compose(pil_img)
            aug_tensor = F.normalize(F.to_tensor(aug_pil), 
                                  DS_MEAN["tiny-imagenet"], 
                                  DS_STD["tiny-imagenet"])
            train_augmented.append(aug_tensor)
        
        # Visualize augmentations
        plt.figure(figsize=(15, 3))
        aug_grid = make_grid(train_augmented, nrow=5, padding=2)
        plt.imshow(denormalize_image(aug_grid).permute(1, 2, 0).cpu())
        plt.title("Augmentation Examples (Same Image)", fontsize=12)
        plt.axis('off')
        plt.savefig("tests/outputs/tiny_imagenet_augmentation_samples.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("✓ Saved augmentation visualization to tests/outputs/tiny_imagenet_augmentation_samples.png")
        
        # Test with PCA labels
        pca_datasets, pca_loaders = get_obj_cls_loader(cfg, pca_labels=True)
        print("\nTesting PCA label integration...")
        
        # Get a batch with PCA labels
        pca_batch = next(iter(pca_loaders["train"]))
        assert len(pca_batch) == 4, "PCA batch should contain (images, pca_labels, orig_labels, img_ids)"
        images, pca_labels, orig_labels, img_ids = pca_batch
        
        # Verify PCA label ranges (should be 0 to 2^n_bits - 1)
        n_classes = 2 ** cfg["n_bits"]
        assert all(0 <= label.item() < n_classes for label in pca_labels), f"PCA labels should be in range [0, {n_classes-1}]"
        print(f"✓ PCA labels are correctly bounded [0, {n_classes-1}]")
        
        # Print some sample PCA mappings
        print("\nSample PCA label mappings:")
        for i in range(min(5, len(pca_labels))):
            print(f"  Image: {img_ids[i]}, Original: {orig_labels[i].item()}, PCA: {pca_labels[i].item()}")
        
        print("\nAll tests passed successfully! 🎉")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_tiny_imagenet_loader() 