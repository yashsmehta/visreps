import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf
import os

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models import utils as model_utils
import visreps.utils as utils
from visreps.utils import calculate_cls_accuracy, rprint, get_logger

class Trainer:
    """Main trainer class for object classification models."""
    def __init__(self, cfg):
        rprint("Initializing trainer...", style="info")
        self.cfg = utils.check_trainer_config(cfg)
        self.setup_environment()
        self.setup_training()
        
    def setup_environment(self):
        """Setup seeds and device"""
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rprint(f"Using device: {self.device}", style="success")
        
    def setup_training(self):
        """Initialize model, data, optimizer"""
        rprint("Setting up training components...", style="setup")
        self.datasets, self.loaders = get_obj_cls_loader(self.cfg)
        self.model = model_utils.load_model(self.cfg, self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        self.optimizer = utils.setup_optimizer(self.model, self.cfg)
        self.scheduler = utils.setup_scheduler(self.optimizer, self.cfg, len(self.loaders["train"]))
        
        # Initialize wandb logging if enabled
        if self.cfg.use_wandb:
            rprint("Initializing W&B logging...", style="highlight")
            self.logger = get_logger(use_wandb=True, log_system_metrics=True, cfg=self.cfg)
            
        # Setup checkpointing if enabled
        if self.cfg.log_checkpoints:
            rprint("Setting up checkpointing...", style="setup")
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)
            rprint(f"Initial checkpoint saved to {self.checkpoint_dir}", style="success")

    @torch.no_grad()
    def evaluate(self, split="test"):
        """Evaluate model on given split"""
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device)

    def train_epoch(self):
        """Train for one epoch with enhanced logging"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.loaders["train"])
        progress_bar = tqdm(self.loaders["train"], desc="Training")
        
        for i, (images, labels) in enumerate(progress_bar):
            # Move data to device and compute forward pass
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            curr_lr = self.scheduler.get_last_lr()[0]
            utils.log_training_step(self.logger, self.cfg, i, loss.item(), curr_lr)
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{curr_lr:.6f}'})
            
        return total_loss / num_batches

    def log_metrics(self, epoch, loss, metrics):
        """Log training metrics"""
        utils.log_training_metrics(self.logger, self.cfg, epoch, loss, metrics, self.scheduler)

    def train(self):
        """Run training loop with enhanced logging"""
        rprint("Starting training...", style="info")
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            # Train and evaluate
            loss = self.train_epoch()
            test_acc = self.evaluate("test")
            metrics = {
                "epoch": epoch,
                "loss": loss,
                "test_acc": test_acc
            }
            
            if self.cfg.evaluate_train and epoch % self.cfg.log_interval == 0:
                metrics["train_acc"] = self.evaluate("train")
                
            # Logging and checkpoints
            self.log_metrics(epoch, loss, metrics)
            
            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                rprint(f"Saving checkpoint at epoch {epoch}...", style="setup")
                model_utils.save_checkpoint(self.checkpoint_dir, epoch, self.model, 
                                         self.optimizer, metrics, self.cfg_dict)

        rprint("Training complete!", style="success")
        if self.cfg.use_wandb:
            wandb.finish()
            
        return self.model