import os
import torch
import pytest
from collections import Counter
from visreps.dataloaders.obj_cls import get_obj_cls_loader

def print_dataset_stats(dataset, split_name):
    """Print statistics about the dataset's labels"""
    all_labels = []
    for _, label in dataset:
        all_labels.append(label)
    
    label_counts = Counter(all_labels)
    print(f"\n{split_name} Set Statistics:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Unique labels: {sorted(label_counts.keys())}")
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count} samples ({count/len(all_labels)*100:.2f}%)")

def test_cifar_pca_labels():
    cfg = {
        "dataset": "cifar10",
        "batchsize": 4,
        "num_workers": 0,  # Use 0 workers for testing
        "data_augment": False
    }
    
    # Test with PCA labels
    datasets, loaders = get_obj_cls_loader(cfg, pca_labels=True)
    
    print("\nTesting CIFAR10 with PCA labels:")
    # Print statistics for both splits
    for split in ['train', 'test']:
        print_dataset_stats(datasets[split], split)
    
    # Verify both train and test sets use PCA labels
    for split in ['train', 'test']:
        loader = loaders[split]
        images, labels = next(iter(loader))
        
        # Check shapes and types
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape == (4, 3, 32, 32)  # batchsize x channels x height x width
        assert labels.shape == (4,)
        
        # Verify labels are within expected range (0-7 for 3 bits)
        assert labels.min() >= 0
        assert labels.max() <= 7, f"Labels in {split} set exceed maximum value for 3 bits"

def test_cifar_original_labels():
    cfg = {
        "dataset": "cifar10",
        "batchsize": 4,
        "num_workers": 0,
        "data_augment": False
    }
    
    # Test without PCA labels
    datasets, loaders = get_obj_cls_loader(cfg, pca_labels=False)
    
    # Verify both train and test sets use original labels
    for split in ['train', 'test']:
        loader = loaders[split]
        images, labels = next(iter(loader))
        
        # Check shapes and types
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape == (4, 3, 32, 32)
        assert labels.shape == (4,)
        
        # Verify labels are within original CIFAR range (0-9)
        assert labels.min() >= 0
        assert labels.max() <= 9, f"Labels in {split} set exceed CIFAR-10 range" 