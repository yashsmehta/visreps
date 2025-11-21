import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
import visreps.models.utils as mutils
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.utils import rprint, get_seed_letter, save_results

logger = logging.getLogger(__name__)

def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (similar to visreps/evals.py)."""
    seed_letter = get_seed_letter(cfg.seed)
    # Construct path to config.json based on conventions
    # Assuming checkpoint_dir points to the experiment root (e.g. model_checkpoints/exp_name)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    rprint(f"Loading config from {path}")
    base = OmegaConf.load(path)
    
    # Override/Merge logic
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    
    # Merge, giving priority to runtime args (cfg)
    return OmegaConf.merge(base, cfg)

class FineTunedModel(nn.Module):
    def __init__(self, feature_extractor, layer_name, input_dim, num_classes=1000):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_name = layer_name
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # feature_extractor returns a dict {layer_name: output}
        features = self.feature_extractor(x)
        out = features[self.layer_name]
        # Flatten if necessary (e.g. conv output)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

def get_feature_dim(model, layer_name, device):
    """Run a dummy pass to determine feature dimension."""
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        features = model(dummy)
    out = features[layer_name]
    return out.view(out.size(0), -1).shape[1]

def verify_freezing(model):
    """Verify that only the classifier parameters are trainable."""
    rprint("\n[Verifying Parameter Freezing]", style="warning")
    trainable = []
    frozen = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    
    # Check that all trainable parameters are in the classifier
    non_classifier_trainable = [name for name in trainable if not name.startswith('classifier')]
    
    if non_classifier_trainable:
        raise RuntimeError(f"ERROR: Found trainable parameters outside classifier: {non_classifier_trainable}")
        
    if not trainable:
         raise RuntimeError("ERROR: No trainable parameters found!")

    rprint(f"✓ Frozen parameters: {len(frozen)} (Base Model)", style="success")
    rprint(f"✓ Trainable parameters: {len(trainable)} (Classifier: {trainable})", style="success")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a linear classifier on a specific layer.")
    # Model loading args
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to experiment directory (e.g. model_checkpoints/my_exp)")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoint_epoch_90.pth", help="Checkpoint filename")
    parser.add_argument("--cfg_id", type=str, required=True, help="Config ID (e.g. 1000 for standard, or pca_n_classes)")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for training (to find correct folder)")
    
    # Finetuning args
    parser.add_argument("--layer", type=str, default="fc1", help="Layer to attach classifier to (e.g. fc1, conv5)")
    # dataset argument removed - hardcoded to imagenet
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()
    
    # Base config
    cfg = OmegaConf.create(vars(args))
    cfg.load_model_from = "checkpoint"
    cfg.return_nodes = [args.layer]
    
    # Load full config
    cfg = _load_cfg(cfg)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rprint(f"Using device: {dev}")

    # 1. Load Base Model
    rprint("\n[1/4] Loading base model...", style="info")
    # We force num_classes=None as we are loading from checkpoint
    base_model = mutils.load_model(cfg, dev)
    
    # 2. Configure Feature Extractor (this attaches hooks)
    rprint(f"\n[2/4] Configuring feature extractor for layer: {args.layer}", style="info")
    feature_extractor = mutils.configure_feature_extractor(cfg, base_model)
    
    # Determine input dimension
    feature_dim = get_feature_dim(feature_extractor, args.layer, dev)
    rprint(f"Feature dimension for {args.layer}: {feature_dim}")

    # 3. Setup Fine-tuning Model
    # Hardcode num_classes=1000 for ImageNet
    model = FineTunedModel(feature_extractor, args.layer, feature_dim, num_classes=1000).to(dev)
    
    # Freeze base parameters
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    # Verify freezing
    verify_freezing(model)

    # Only classifier requires grad
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rprint(f"Total Trainable parameters (Linear Layer): {trainable_params}")

    # 4. Data Loading
    rprint("\n[3/4] Loading Data...", style="info")
    # Ensure cfg has necessary dataset keys
    # Hardcode dataset to imagenet and disable PCA labels
    cfg.dataset = "imagenet" 
    cfg.pca_labels = False 
    
    datasets, loaders = get_obj_cls_loader(cfg, shuffle=True, preprocess=True, train_test_split=True)
    train_loader = loaders['train']
    val_loader = loaders['test']
    
    # 5. Training Loop
    rprint("\n[4/4] Starting Fine-tuning...", style="info")
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(dev), targets.to(dev)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(dev), targets.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        rprint(f"Epoch {epoch+1} Result | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", style="success")

if __name__ == "__main__":
    main()

