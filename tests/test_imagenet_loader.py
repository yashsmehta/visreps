import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Dict
from PIL import Image

from visreps.dataloaders.obj_cls import get_obj_cls_loader, DS_MEAN, DS_STD

def denormalize_image(img: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    """Denormalize image tensor for visualization"""
    img = img.clone()
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
    
    # Flatten axes for easier iteration
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

def test_imagenet_loader():
    """Test ImageNet dataloader functionality"""
    print("Testing ImageNet dataloader...")
    
    # Configuration
    cfg = {
        "dataset": "imagenet",
        "dataset_path": "/data/shared/datasets/imagenet",
        "batchsize": 32,
        "num_workers": 4,
        "data_augment": True,
        "train_ratio": 0.8  # 80-20 train-test split
    }
    
    try:
        # Load datasets and dataloaders
        datasets, loaders = get_obj_cls_loader(cfg)
        print("✓ Successfully created datasets and dataloaders")
        
        # Basic checks
        assert "train" in loaders and "test" in loaders, "Missing train or test splits"
        print("✓ Both train and test splits present")
        
        # Verify label consistency
        train_labels = set(label for _, label, img_id in datasets["train"].samples)
        test_labels = set(label for _, label, img_id in datasets["test"].samples)
        all_labels = train_labels.union(test_labels)
        assert all(isinstance(label, int) for label in all_labels), "All labels should be integers"
        assert max(all_labels) < 1000 and min(all_labels) >= 0, "Labels should be in range [0, 999]"
        print(f"✓ Found {len(all_labels)} unique labels")
        
        # Test batch loading
        train_batch = next(iter(loaders["train"]))
        test_batch = next(iter(loaders["test"]))
        
        # Check shapes
        assert len(train_batch) == 2, "Batch should contain (images, labels)"
        images, labels = train_batch
        assert images.shape == (cfg["batchsize"], 3, 224, 224), f"Unexpected image shape: {images.shape}"
        assert labels.shape == (cfg["batchsize"],), f"Unexpected labels shape: {labels.shape}"
        print("✓ Batch shapes are correct")
        
        # Check value ranges
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.long, "Labels should be long"
        assert all(0 <= label.item() < 1000 for label in labels), "Labels should be in range [0, 999]"
        print("✓ Data types are correct")
        
        # Print some sample labels
        print("\nSample labels in batch:")
        for i in range(min(5, len(labels))):
            label_idx = labels[i].item()
            print(f"  Class {label_idx}")
        
        # Visualize samples
        os.makedirs("tests/outputs", exist_ok=True)
        visualize_batch(images, labels, save_path="tests/outputs/imagenet_samples.png")
        print("✓ Saved sample visualization to tests/outputs/imagenet_samples.png")
        
        # Test data augmentation (train vs test transforms)
        test_img = test_batch[0][0]  # First image from test batch
        pil_img = tensor_to_pil(test_img)
        train_augmented = []
        
        # Get just the augmentation transforms (skip normalization)
        aug_transforms = [t for t in datasets["train"].transform.transforms 
                        if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomRotation))]
        aug_compose = transforms.Compose(aug_transforms)
        
        # Apply augmentations and convert back to normalized tensor
        for _ in range(5):
            aug_pil = aug_compose(pil_img)
            aug_tensor = F.normalize(F.to_tensor(aug_pil), 
                                  DS_MEAN["imgnet"], 
                                  DS_STD["imgnet"])
            train_augmented.append(aug_tensor)
        
        # Visualize augmentations
        plt.figure(figsize=(15, 3))
        aug_grid = make_grid(train_augmented, nrow=5, padding=2)
        plt.imshow(denormalize_image(aug_grid).permute(1, 2, 0).cpu())
        plt.title("Augmentation Examples (Same Image)", fontsize=12)
        plt.axis('off')
        plt.savefig("tests/outputs/augmentation_samples.png", bbox_inches='tight', dpi=150)
        plt.show()
        print("✓ Saved augmentation visualization to tests/outputs/augmentation_samples.png")
        
        print("\nAll tests passed successfully! 🎉")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_imagenet_loader() 