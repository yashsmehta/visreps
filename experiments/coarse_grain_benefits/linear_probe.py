"""
Linear probe evaluation via pre-extracted features with SRP dimensionality reduction.
"""
import argparse, csv, os, re, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

import visreps.models.utils as mutils
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.utils import rprint, get_seed_letter
from visreps.analysis.sparse_random_projection import get_srp_transformer

SRP_DIM = 4096


def _load_cfg(cfg):
    """Merge runtime cfg with training cfg."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(path)
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


def _build_srp_matrices(model, loader, layers, device):
    """Build SRP projection matrices for each layer."""
    probe_imgs, _ = next(iter(loader))
    with torch.no_grad():
        probe_feats = model(probe_imgs.to(device))
    
    srp = {}
    for layer in layers:
        D = probe_feats[layer].view(probe_imgs.size(0), -1).size(1)
        k = min(SRP_DIM, D)
        transformer = get_srp_transformer(D=D, k=k, density=None, seed=42, 
                                          cache_dir="model_checkpoints/srp_cache")
        if transformer:
            coo = transformer.components_.tocoo()
            indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
            values = torch.from_numpy(coo.data).float()
            srp[layer] = torch.sparse_coo_tensor(indices, values, coo.shape).to(device)
            rprint(f"  SRP {layer}: {D}→{k}", style="info")
    return srp


def extract_features(model, loader, layers, device, cache_dir, split, srp=None):
    """Extract features with SRP projection, save to disk (memory-efficient)."""
    cache_paths = {l: os.path.join(cache_dir, l, f"{split}_srp{SRP_DIM}.pt") for l in layers}
    
    if all(os.path.exists(p) for p in cache_paths.values()):
        rprint(f"Cached {split} features exist", style="info")
        return cache_paths, srp
    
    if srp is None:
        srp = _build_srp_matrices(model, loader, layers, device)
    
    rprint(f"Extracting {split} features (SRP→{SRP_DIM})", style="info")
    all_feats = {l: [] for l in layers}
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting {split}"):
            feats = model(imgs.to(device))
            for layer in layers:
                out = feats[layer].view(imgs.size(0), -1)
                if layer in srp:
                    out = torch.sparse.mm(srp[layer], out.t()).t()
                all_feats[layer].append(out.cpu())
            all_labels.append(labels)
    
    labels_tensor = torch.cat(all_labels)
    for layer in layers:
        features = torch.cat(all_feats[layer])
        os.makedirs(os.path.dirname(cache_paths[layer]), exist_ok=True)
        torch.save({'features': features, 'labels': labels_tensor}, cache_paths[layer])
        rprint(f"  Saved {layer}: {features.shape}", style="success")
        del features
    del all_feats, all_labels, labels_tensor
    
    return cache_paths, srp


def train_probe(train_loader, val_loader, feat_dim, n_classes, device, epochs, lr):
    """Train linear classifier on pre-extracted features."""
    clf = nn.Linear(feat_dim, n_classes).to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        clf.train()
        correct, total = 0, 0
        for feats, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            feats, targets = feats.to(device), targets.to(device)
            opt.zero_grad()
            loss = loss_fn(clf(feats), targets)
            loss.backward()
            opt.step()
            correct += (clf(feats).argmax(1) == targets).sum().item()
            total += targets.size(0)
        
        clf.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                val_correct += (clf(feats).argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        
        train_acc, val_acc = 100.*correct/total, 100.*val_correct/val_total
        rprint(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%", style="success")
    
    return train_acc, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="/data/ymehta3/alexnet_pca")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoint_epoch_20.pth")
    parser.add_argument("--cfg_id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layers", type=str, nargs='+', default=["fc1"])
    parser.add_argument("--batchsize", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache_dir", type=str, default="feature_cache")
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--results_csv", type=str, default="experiments/coarse_grain_benefits/results/linear_probe_results.csv")
    args = parser.parse_args()
    
    cfg = OmegaConf.create(vars(args))
    cfg.load_model_from = "checkpoint"
    cfg.return_nodes = args.layers
    cfg = _load_cfg(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"Device: {device} | SRP→{SRP_DIM}")
    
    # Load model
    model = mutils.configure_feature_extractor(cfg, mutils.load_model(cfg, device))
    
    # Load data
    data_cfg = OmegaConf.create({**OmegaConf.to_container(cfg), 
                                  "dataset": "imagenet", "pca_labels": False, 
                                  "data_augment": False, "batchsize": 256})
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, preprocess=True, train_test_split=True)
    
    # Extract features (saves to disk, doesn't hold in memory)
    seed_letter = get_seed_letter(cfg.seed)
    cache_base = os.path.join(args.cache_dir, f"cfg{cfg.cfg_id}{seed_letter}")
    train_paths, srp = extract_features(model, loaders['train'], args.layers, device, cache_base, "train")
    val_paths, _ = extract_features(model, loaders['test'], args.layers, device, cache_base, "val", srp)
    del model, srp  # free GPU memory before probe training
    torch.cuda.empty_cache()
    
    # Prepare CSV
    epoch = int(re.search(r'epoch_(\d+)', args.checkpoint_model).group(1)) if re.search(r'epoch_(\d+)', args.checkpoint_model) else -1
    csv_exists = os.path.exists(args.results_csv)
    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['checkpoint_dir', 'cfg_id', 'checkpoint_model', 'epoch', 'train_acc', 'test_acc', 'layer'])
        
        # Train probes (load one layer at a time)
        for layer in args.layers:
            rprint(f"\nTraining probe: {layer}", style="info")
            train_data = torch.load(train_paths[layer], weights_only=True)
            val_data = torch.load(val_paths[layer], weights_only=True)
            
            train_loader = DataLoader(TensorDataset(train_data['features'], train_data['labels']), batch_size=args.batchsize, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_data['features'], val_data['labels']), batch_size=args.batchsize)
            
            train_acc, test_acc = train_probe(train_loader, val_loader, train_data['features'].shape[1], 1000, device, args.epochs, args.lr)
            writer.writerow([args.checkpoint_dir, cfg.cfg_id, args.checkpoint_model, epoch, f"{train_acc:.2f}", f"{test_acc:.2f}", layer])
            f.flush()
            rprint(f"✓ {layer}: Train={train_acc:.2f}%, Test={test_acc:.2f}%", style="success")
            del train_data, val_data  # free memory before next layer
    
    if not args.no_cleanup:
        shutil.rmtree(cache_base, ignore_errors=True)
    
    rprint(f"\nResults saved to {args.results_csv}", style="success")


if __name__ == "__main__":
    main()
