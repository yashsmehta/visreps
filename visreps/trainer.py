import torch
import wandb
import torch.nn as nn
from tqdm import tqdm

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
            # Add train loader length to config for fractional epoch calculation
            self.cfg.train_loader_len = len(self.loaders["train"])
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

    def train_epoch(self, epoch):
        """Train for one epoch with per-batch logging"""
        self.model.train()
        num_batches = len(self.loaders["train"])
        progress_bar = tqdm(self.loaders["train"], desc="Training")
        last_loss = None
        
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
            last_loss = loss.item()
            utils.log_training_step(self.logger, self.cfg, epoch, i, last_loss, curr_lr)
            
            # Update progress
            progress_bar.set_postfix({'loss': f'{last_loss:.4f}', 'lr': f'{curr_lr:.6f}'})
        
        return last_loss

    def log_metrics(self, epoch, loss, metrics):
        """Log training metrics"""
        utils.log_training_metrics(self.logger, self.cfg, epoch, loss, metrics, self.scheduler)

    def train(self):
        """Run training loop with enhanced logging"""
        rprint("Starting training...", style="info")
        
        for epoch in range(1, self.cfg.num_epochs + 1):
            # Train and evaluate
            epoch_loss = self.train_epoch(epoch)
            test_acc = self.evaluate("test")
            metrics = {
                "epoch": epoch,
                "test_acc": test_acc
            }
            
            if self.cfg.evaluate_train and epoch % self.cfg.log_interval == 0:
                metrics["train_acc"] = self.evaluate("train")
                
            # Logging and checkpoints
            self.log_metrics(epoch, epoch_loss, metrics)
            
            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                rprint(f"Saving checkpoint at epoch {epoch}...", style="setup")
                model_utils.save_checkpoint(self.checkpoint_dir, epoch, self.model, 
                                         self.optimizer, metrics, self.cfg_dict)

        rprint("Training complete!", style="success")
        if self.cfg.use_wandb:
            wandb.finish()
            
        return self.model