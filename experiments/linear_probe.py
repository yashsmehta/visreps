"""
Linear probe evaluation via pre-extracted features.
Extracts features once to disk, trains linear classifier on GPU, optionally cleans up.
"""
import argparse
import csv
import os
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

import visreps.models.utils as mutils
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.utils import rprint, get_seed_letter


def _load_cfg(cfg):
    """Merge runtime cfg with training cfg."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    rprint(f"Loading config from {path}")
    base = OmegaConf.load(path)
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


def extract_all_layers_features(model, loader, layers, device, cache_dir, split_name):
    """Extract features from ALL layers in a single pass, save each to disk."""
    # Check if all layers already cached
    cache_paths = {layer: os.path.join(cache_dir, layer, f"{split_name}.pt") for layer in layers}
    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    
    layers_str = ", ".join(layers)
    if all_cached:
        rprint(f"Loading cached {split_name} features: [{layers_str}]", style="info")
        return {
            layer: torch.load(path, weights_only=True)
            for layer, path in cache_paths.items()
        }
    
    rprint(f"Extracting {split_name} features: [{layers_str}]", style="info")
    
    # Accumulators for each layer
    all_features = {layer: [] for layer in layers}
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting {split_name}"):
            imgs = imgs.to(device)
            feats = model(imgs)
            for layer in layers:
                out = feats[layer].view(imgs.size(0), -1)  # Flatten
                all_features[layer].append(out.cpu())
            all_labels.append(labels)
    
    labels_tensor = torch.cat(all_labels, dim=0)
    
    # Save each layer to its own file
    results = {}
    for layer in layers:
        features_tensor = torch.cat(all_features[layer], dim=0)
        cache_path = cache_paths[layer]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = {'features': features_tensor, 'labels': labels_tensor}
        torch.save(data, cache_path)
        results[layer] = data
        rprint(f"  Saved {layer}: {features_tensor.shape}", style="success")
    
    return results


def create_feature_loader(features, labels, batch_size, shuffle=True):
    """Create DataLoader from pre-extracted features."""
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=0, pin_memory=True)


def train_linear_probe(train_loader, val_loader, feature_dim, num_classes, 
                       device, epochs=10, lr=1e-3):
    """Train linear classifier on pre-extracted features (all on GPU)."""
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        
        train_acc = 100. * correct / total
        
        # Validation
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        rprint(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%", style="success")
    
    return classifier, train_acc, val_acc


def main():
    parser = argparse.ArgumentParser(description="Linear probe on pre-extracted features.")
    # Model loading
    # parser.add_argument("--checkpoint_dir", type=str, default="/data/ymehta3/alexnet_pca")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/ymehta3/imagenet1k")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoint_epoch_20.pth")
    parser.add_argument("--cfg_id", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    
    # Probe settings
    parser.add_argument("--layers", type=str, nargs='+', default=["conv4", "fc1", "fc2"], 
                        help="Layer(s) to extract features from")
    parser.add_argument("--batchsize", type=int, default=4096, help="Large batch OK since no CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Cache control
    parser.add_argument("--cache_dir", type=str, default="feature_cache")
    parser.add_argument("--no_cleanup", action="store_true", help="Keep cached features after training")
    
    # Output
    parser.add_argument("--results_csv", type=str, default="linear_probe_results.csv")
    
    args = parser.parse_args()
    
    # Setup
    cfg = OmegaConf.create(vars(args))
    cfg.load_model_from = "checkpoint"
    cfg.return_nodes = args.layers
    cfg = _load_cfg(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"Using device: {device}")
    
    # Extract epoch from checkpoint name (e.g., "checkpoint_epoch_20.pth" → 20)
    epoch_match = re.search(r'epoch_(\d+)', args.checkpoint_model)
    checkpoint_epoch = int(epoch_match.group(1)) if epoch_match else -1
    
    # 1. Load model & configure feature extractor
    rprint("\n[1/4] Loading model...", style="info")
    base_model = mutils.load_model(cfg, device)
    feature_extractor = mutils.configure_feature_extractor(cfg, base_model)
    
    # 2. Get data loaders (for extraction)
    rprint("\n[2/4] Loading data...", style="info")
    extraction_cfg = OmegaConf.create({
        **OmegaConf.to_container(cfg),
        "dataset": "imagenet",
        "pca_labels": False,
        "data_augment": False,
        "batchsize": 256,
    })
    _, loaders = get_obj_cls_loader(extraction_cfg, shuffle=False, preprocess=True, train_test_split=True)
    
    # Prepare CSV output
    csv_exists = os.path.exists(args.results_csv)
    csv_file = open(args.results_csv, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['checkpoint_dir', 'cfg_id', 'checkpoint_model', 'epoch', 
                            'train_acc', 'test_acc', 'layer'])
    
    seed_letter = get_seed_letter(cfg.seed)
    cache_base = os.path.join(args.cache_dir, f"cfg{cfg.cfg_id}{seed_letter}")
    results = []
    
    # 3. Extract features
    train_data = extract_all_layers_features(
        feature_extractor, loaders['train'], args.layers, device, cache_base, "train"
    )
    val_data = extract_all_layers_features(
        feature_extractor, loaders['test'], args.layers, device, cache_base, "val"
    )
    
    # 4. Train linear probes
    for i, layer in enumerate(args.layers):
        rprint(f"\nTraining probe {i+1}/{len(args.layers)}: {layer}", style="info")
        
        train_features = train_data[layer]['features']
        train_labels = train_data[layer]['labels']
        val_features = val_data[layer]['features']
        val_labels = val_data[layer]['labels']
        
        feature_dim = train_features.shape[1]
        rprint(f"Feature dim: {feature_dim}, Train: {len(train_features)}, Val: {len(val_features)}")
        
        # Create feature loaders
        train_loader = create_feature_loader(train_features, train_labels, args.batchsize, shuffle=True)
        val_loader = create_feature_loader(val_features, val_labels, args.batchsize, shuffle=False)
        
        # Train linear probe
        classifier, train_acc, test_acc = train_linear_probe(
            train_loader, val_loader, feature_dim, num_classes=1000,
            device=device, epochs=args.epochs, lr=args.lr
        )
        
        # Record result
        row = [args.checkpoint_dir, cfg.cfg_id, args.checkpoint_model, 
               checkpoint_epoch, f"{train_acc:.2f}", f"{test_acc:.2f}", layer]
        csv_writer.writerow(row)
        csv_file.flush()
        results.append((layer, train_acc, test_acc))
        
        rprint(f"✓ {layer}: Train={train_acc:.2f}%, Test={test_acc:.2f}%", style="success")
    
    # Cleanup all cached features
    if not args.no_cleanup:
        shutil.rmtree(cache_base, ignore_errors=True)
    
    csv_file.close()
    
    # Summary
    rprint(f"\nResults saved to {args.results_csv}", style="success")
    for layer, train_acc, test_acc in results:
        rprint(f"  {layer}: Train={train_acc:.2f}%, Test={test_acc:.2f}%")


if __name__ == "__main__":
    main()
