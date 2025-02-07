import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models import utils as model_utils
import visreps.utils as utils
from visreps.utils import calculate_cls_accuracy, rprint, get_logger


class Trainer:
    """Main trainer class for object classification models."""
    def __init__(self, cfg):
        rprint("Initializing trainer...", style="info")
        self.cfg = utils.check_trainer_config(cfg)
        # Initialize logger to None before setting up training components.
        self.logger = None
        self.setup_environment()
        self.setup_training()

    def setup_environment(self):
        """Set up seeds and device."""
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rprint(f"Using device: {self.device}", style="success")

    def setup_training(self):
        """Initialize model, data, optimizer, scheduler, logger and checkpointing."""
        rprint("Setting up training components...", style="setup")
        self.datasets, self.loaders = get_obj_cls_loader(self.cfg)
        
        # Get number of classes for the model
        num_classes = self.cfg.pca_n_classes if self.cfg.pca_labels else self.datasets["train"].num_classes
        
        # Initialize model with correct number of classes
        self.model = model_utils.load_model(self.cfg, self.device, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer and scheduler.
        self.optimizer = utils.setup_optimizer(self.model, self.cfg)
        self.scheduler = utils.setup_scheduler(self.optimizer, self.cfg, len(self.loaders["train"]))

        # Initialize wandb logging if enabled.
        if self.cfg.use_wandb:
            # Store train loader length in the config for fractional epoch calculation.
            self.cfg.train_loader_len = len(self.loaders["train"])
            rprint("Initializing W&B logging...", style="highlight")
            self.logger = get_logger(use_wandb=True, log_system_metrics=True, cfg=self.cfg)

        # Setup checkpointing if enabled.
        if self.cfg.log_checkpoints:
            rprint("Setting up checkpointing...", style="setup")
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)
            rprint(f"Initial checkpoint saved to {self.checkpoint_dir}", style="success")
            
        # Store number of classes for metrics tracking
        self._num_classes = num_classes

    @torch.no_grad()
    def evaluate(self, split="test"):
        """Evaluate the model on the specified data split."""
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device)

    def train_epoch(self, epoch):
        """Train the model for one epoch with per-batch logging."""
        self.model.train()
        progress_bar = tqdm(self.loaders["train"], desc=f"Epoch {epoch}")
        last_loss = 0.0
        
        # Track detailed metrics
        epoch_stats = defaultdict(float)  # Changed to float for accumulation
        correct = 0
        total = 0
        
        # Debug first few batches in detail
        debug_batches = 2 if epoch == 1 else 0
        
        for i, (images, labels) in enumerate(progress_bar):
            # Detailed batch inspection for first few batches
            if debug_batches > 0:
                print(f"\nDetailed batch {i+1} inspection:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels: {labels.tolist()}")
                print(f"Unique labels: {torch.unique(labels).tolist()}")
                print(f"Image stats:")
                print(f"  Range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"  Mean: {images.mean():.3f}")
                print(f"  Std: {images.std():.3f}")
                print(f"Per-channel stats:")
                for c in range(3):
                    print(f"  Channel {c}:")
                    print(f"    Range: [{images[:,c].min():.3f}, {images[:,c].max():.3f}]")
                    print(f"    Mean: {images[:,c].mean():.3f}")
                    print(f"    Std: {images[:,c].std():.3f}")
                debug_batches -= 1
            
            # Move data to the configured device
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Debug model outputs for first few batches
            if debug_batches == 1:
                print(f"\nModel output inspection:")
                print(f"Output shape: {outputs.shape}")
                print(f"Output stats:")
                print(f"  Range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                print(f"  Mean: {outputs.mean():.3f}")
                print(f"  Std: {outputs.std():.3f}")
                print(f"Output class distribution:")
                pred_classes = outputs.argmax(dim=1)
                unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
                print(f"  Predicted classes: {unique_preds.tolist()}")
                print(f"  Class counts: {pred_counts.tolist()}")
            
            loss = self.criterion(outputs, labels)

            # Track predictions and accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batch_acc = (preds == labels).float().mean().item()
            
            # Track statistics
            with torch.no_grad():
                epoch_stats['batch_loss'] += loss.item()
                epoch_stats['batch_accuracy'] += batch_acc
                epoch_stats['output_mean'] += outputs.mean().item()
                epoch_stats['output_std'] += outputs.std().item()
                epoch_stats['n_batches'] = i + 1
                
                # Track per-class accuracy
                for c in range(self._num_classes):
                    mask = (labels == c)
                    if mask.any():
                        correct_pred = (preds[mask] == labels[mask]).float().sum().item()
                        total_pred = mask.sum().item()
                        epoch_stats[f'class_{c}_correct'] += correct_pred
                        epoch_stats[f'class_{c}_total'] += total_pred

            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping if configured
            if hasattr(self.cfg, 'grad_clip') and self.cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            else:
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None]))
            
            epoch_stats['grad_norm'] += grad_norm.item()
            
            self.optimizer.step()

            last_loss = loss.item()
            curr_lr = self.optimizer.param_groups[0]['lr']  # Get current learning rate directly
            epoch_stats['learning_rate'] = curr_lr  # Just keep the last one

            # Update progress bar with more metrics
            progress_bar.set_postfix({
                'loss': f'{last_loss:.4f}',
                'acc': f'{(correct/total)*100:.2f}%',
                'lr': f'{curr_lr:.6f}',
                'grad_norm': f'{grad_norm:.4f}'
            })

        # Compute epoch statistics
        n_batches = epoch_stats['n_batches']
        epoch_metrics = {
            'epoch_loss': epoch_stats['batch_loss'] / n_batches,
            'epoch_accuracy': correct / total,
            'epoch_grad_norm': epoch_stats['grad_norm'] / n_batches,
            'epoch_output_mean': epoch_stats['output_mean'] / n_batches,
            'epoch_output_std': epoch_stats['output_std'] / n_batches,
            'learning_rate': curr_lr
        }
        
        # Compute per-class accuracies
        for c in range(self._num_classes):
            if epoch_stats[f'class_{c}_total'] > 0:
                acc = epoch_stats[f'class_{c}_correct'] / epoch_stats[f'class_{c}_total']
                epoch_metrics[f'class_{c}_accuracy'] = acc

        if self.logger is not None:
            wandb.log(epoch_metrics)

        return last_loss, epoch_metrics

    def log_metrics(self, epoch, loss, metrics):
        """Log aggregated training metrics."""
        if self.logger is not None:
            combined_metrics = {
                'epoch': metrics['epoch'],
                'test/accuracy': metrics['test_acc'],
                'train/loss': loss,
            }
            
            if 'train_acc' in metrics:
                combined_metrics['train/accuracy'] = metrics['train_acc']
                
            if 'epoch_metrics' in metrics:
                for k, v in metrics['epoch_metrics'].items():
                    combined_metrics[f'train/{k}'] = v
                    
            utils.log_training_metrics(self.logger, self.cfg, epoch, loss, combined_metrics, self.scheduler)

    def train(self):
        """Run the complete training loop with logging and checkpointing."""
        rprint("Starting training...", style="info")

        for epoch in range(1, self.cfg.num_epochs + 1):
            # Train for one epoch
            epoch_loss, epoch_metrics = self.train_epoch(epoch)

            # Step the scheduler after each epoch without passing epoch parameter
            self.scheduler.step()

            # Evaluate on the test split
            test_acc = self.evaluate("test")
            metrics = {
                "epoch": epoch,
                "test_acc": test_acc,
                "epoch_metrics": epoch_metrics
            }

            # Optionally evaluate on the training split
            if self.cfg.evaluate_train and epoch % self.cfg.log_interval == 0:
                metrics["train_acc"] = self.evaluate("train")

            # Log aggregated metrics
            self.log_metrics(epoch, epoch_loss, metrics)

            # Save a checkpoint if enabled
            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                rprint(f"Saving checkpoint at epoch {epoch}...", style="setup")
                model_utils.save_checkpoint(
                    self.checkpoint_dir, epoch, self.model, self.optimizer, metrics, self.cfg_dict
                )

        rprint("Training complete!", style="success")
        if self.cfg.use_wandb:
            wandb.finish()

        return self.model