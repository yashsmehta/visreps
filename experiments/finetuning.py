"""
Linear probe evaluation via pre-extracted features.
Extracts features once to disk, trains linear classifier on GPU, optionally cleans up.
"""
import argparse
import os
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


def extract_and_cache_features(model, loader, layer, device, cache_path):
    """Extract features for entire dataset and save to disk."""
    if os.path.exists(cache_path):
        rprint(f"Loading cached features from {cache_path}", style="info")
        data = torch.load(cache_path, weights_only=True)
        return data['features'], data['labels']
    
    rprint(f"Extracting features to {cache_path}...", style="info")
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            feats = model(imgs)
            out = feats[layer].view(imgs.size(0), -1)  # Flatten
            all_features.append(out.cpu())
            all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Save to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({'features': features, 'labels': labels}, cache_path)
    rprint(f"Saved {len(features)} samples to {cache_path}", style="success")
    
    return features, labels


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
    
    return classifier, val_acc


def main():
    parser = argparse.ArgumentParser(description="Linear probe on pre-extracted features.")
    # Model loading
    parser.add_argument("--checkpoint_dir", type=str, default="/data/ymehta3/alexnet_pca")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoint_epoch_20.pth")
    parser.add_argument("--cfg_id", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    
    # Probe settings
    parser.add_argument("--layer", type=str, default="fc1")
    parser.add_argument("--batchsize", type=int, default=4096, help="Large batch OK since no CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Cache control
    parser.add_argument("--cache_dir", type=str, default="feature_cache")
    parser.add_argument("--no_cleanup", action="store_true", help="Keep cached features after training")
    
    args = parser.parse_args()
    
    # Setup
    cfg = OmegaConf.create(vars(args))
    cfg.load_model_from = "checkpoint"
    cfg.return_nodes = [args.layer]
    cfg = _load_cfg(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"Using device: {device}")
    
    # 1. Load model & configure feature extractor
    rprint("\n[1/5] Loading model...", style="info")
    base_model = mutils.load_model(cfg, device)
    feature_extractor = mutils.configure_feature_extractor(cfg, base_model)
    
    # 2. Get data loaders (for extraction)
    rprint("\n[2/5] Loading data...", style="info")
    cfg.dataset = "imagenet"
    cfg.pca_labels = False
    # Use smaller batch for extraction (images go through CNN)
    extraction_cfg = OmegaConf.create({**OmegaConf.to_container(cfg), "batchsize": 256})
    _, loaders = get_obj_cls_loader(extraction_cfg, shuffle=False, preprocess=True, train_test_split=True)
    
    # 3. Extract features to disk
    rprint("\n[3/5] Extracting features...", style="info")
    seed_letter = get_seed_letter(cfg.seed)
    cache_base = os.path.join(args.cache_dir, f"cfg{cfg.cfg_id}{seed_letter}", args.layer)
    
    train_features, train_labels = extract_and_cache_features(
        feature_extractor, loaders['train'], args.layer, device,
        os.path.join(cache_base, "train.pt")
    )
    val_features, val_labels = extract_and_cache_features(
        feature_extractor, loaders['test'], args.layer, device,
        os.path.join(cache_base, "val.pt")
    )
    
    feature_dim = train_features.shape[1]
    rprint(f"Feature dim: {feature_dim}, Train: {len(train_features)}, Val: {len(val_features)}")
    
    # 4. Create feature loaders (fast—no CNN, just tensors)
    rprint("\n[4/5] Creating feature loaders...", style="info")
    train_loader = create_feature_loader(train_features, train_labels, args.batchsize, shuffle=True)
    val_loader = create_feature_loader(val_features, val_labels, args.batchsize, shuffle=False)
    
    # 5. Train linear probe
    rprint("\n[5/5] Training linear probe...", style="info")
    classifier, final_acc = train_linear_probe(
        train_loader, val_loader, feature_dim, num_classes=1000,
        device=device, epochs=args.epochs, lr=args.lr
    )
    
    rprint(f"\n✓ Final validation accuracy: {final_acc:.2f}%", style="success")
    
    # Cleanup by default (use --no_cleanup to keep)
    if not args.no_cleanup:
        rprint(f"Cleaning up cache at {cache_base}...", style="warning")
        shutil.rmtree(cache_base, ignore_errors=True)
        rprint("Cache deleted.", style="success")


if __name__ == "__main__":
    main()
