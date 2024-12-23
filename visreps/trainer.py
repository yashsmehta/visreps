import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf

from visreps.dataloaders.obj_cls import get_obj_cls_loader
import visreps.utils as utils
from visreps.models import utils as model_utils
from visreps.metrics import calculate_cls_accuracy
from visreps.utils import rprint

class Trainer:
    """Main trainer class for object classification models.
    
    Usage:
        trainer = Trainer(cfg)
        model = trainer.train()
    """
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
        optimizer_cls = utils.get_optimizer_class(self.cfg.optimizer)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.cfg.learning_rate)
        
        if self.cfg.use_wandb:
            rprint("Initializing W&B logging...", style="highlight")
            self._init_wandb()
            
        if self.cfg.log_checkpoints:
            rprint("Setting up checkpointing...", style="setup")
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)
            rprint(f"Initial checkpoint saved to {self.checkpoint_dir}", style="success")

    def _init_wandb(self):
        """Initialize W&B logging"""
        wandb.init(
            project=self.cfg.exp_name,
            group=self.cfg.group,
            name=f"seed_{self.cfg.seed}",
            config=OmegaConf.to_container(self.cfg, resolve=True)
        )

    @torch.no_grad()
    def evaluate(self, split="test"):
        """Evaluate model on given split"""
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(self.loaders["train"], desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loaders["train"])

    def log_metrics(self, epoch, loss, metrics):
        """Log metrics to console and W&B"""
        if self.cfg.use_wandb:
            wandb.log(metrics)
            
        if epoch % self.cfg.log_interval == 0 or epoch == self.cfg.num_epochs:
            rprint(
                f"Epoch [{epoch}/{self.cfg.num_epochs}]",
                f"Loss: {loss:.6f}",
                f"Test Acc: {metrics['test_acc']:.2f}%",
                style="info"
            )
            if 'train_acc' in metrics:
                rprint(f"Train Acc: {metrics['train_acc']:.2f}%", style="success")

    def train(self):
        """Run the main training loop and return the trained model."""
        rprint("Starting training...", style="info")
        for epoch in range(1, self.cfg.num_epochs + 1):
            # Train and evaluate
            loss = self.train_epoch()
            metrics = {
                "epoch": epoch,
                "loss": loss,
                "test_acc": self.evaluate("test")
            }
            if self.cfg.evaluate_train:
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