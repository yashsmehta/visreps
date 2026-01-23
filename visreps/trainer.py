import time
import torch
import torch.nn as nn
from tqdm import tqdm

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models import utils as model_utils
import visreps.utils as utils
from visreps.utils import calculate_cls_accuracy, is_interactive_environment, MetricsLogger


class Trainer:
    """Trainer class for object classification models."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()

    def _setup(self):
        # Environment setup
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Data and model setup
        self.datasets, self.loaders = get_obj_cls_loader(self.cfg)
        num_classes = self.cfg.pca_n_classes if self.cfg.pca_labels else self.datasets["train"].num_classes
        self.model = model_utils.load_model(self.cfg, self.device, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = utils.setup_optimizer(self.model, self.cfg)
        self.scheduler = utils.setup_scheduler(self.optimizer, self.cfg)

        # Logging and checkpointing
        self.checkpoint_dir = None
        self.cfg_dict = None
        if self.cfg.log_checkpoints:
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)

        self.metrics_logger = MetricsLogger(self.cfg, self.checkpoint_dir)

    @torch.no_grad()
    def evaluate(self, split="test"):
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device)

    def _compute_gradients(self, loss):
        """Compute gradients and optionally clip them."""
        loss.backward()
        if hasattr(self.cfg, 'grad_clip') and self.cfg.grad_clip > 0:
            return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        return torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None]))

    def _create_progress_bar(self, loader, epoch):
        """Create progress bar for interactive environments."""
        if is_interactive_environment():
            return tqdm(loader, desc=f"Epoch {epoch}", leave=False), True
        print(f"Starting Epoch {epoch}")
        return loader, False

    def train_epoch(self, epoch):
        self.model.train()
        loader, use_pbar = self._create_progress_bar(self.loaders["train"], epoch)

        total_loss = 0
        total_grad_norm = 0

        for i, (images, labels) in enumerate(loader):
            # Forward pass
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)

            # Backward pass
            grad_norm = self._compute_gradients(loss)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            lr = self.optimizer.param_groups[0]['lr']

            # Logging
            self.metrics_logger.log_training_step(epoch, i, loss.item(), lr)

            # Progress updates (only for interactive environments)
            if use_pbar:
                avg_loss = total_loss / (i + 1)
                loader.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'LR': f'{lr:.6f}', 'Grad Norm': f'{grad_norm:.4f}'})

        avg_loss = total_loss / len(loader)
        return avg_loss, {'epoch_loss': avg_loss, 'learning_rate': lr}

    def train(self):
        start_time = time.time()
        for epoch in range(1, self.cfg.num_epochs + 1):
            epoch_loss, epoch_metrics = self.train_epoch(epoch)
            self.scheduler.step()

            metrics = {"epoch": epoch, "epoch_metrics": epoch_metrics}

            # Print ETA after first epoch
            if epoch == 1 and is_interactive_environment():
                eta_seconds = (time.time() - start_time) * (self.cfg.num_epochs - 1)
                hours, minutes = int(eta_seconds // 3600), int((eta_seconds % 3600) // 60)
                time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                print(f"⏳ Estimated time remaining: {time_str}")

            # Periodic evaluation and checkpointing
            if epoch % self.cfg.log_interval == 0:
                # Evaluate on both splits
                for split in ["test", "train"]:
                    top1, top5 = self.evaluate(split)
                    metrics[f"{split}_acc"] = top1
                    metrics[f"{split}_top5"] = top5

                self.metrics_logger.log_metrics(epoch, epoch_loss, metrics)

            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                model_utils.save_checkpoint(
                    self.checkpoint_dir, epoch, self.model, self.optimizer, metrics, self.cfg_dict
                )

        self.metrics_logger.finish()
        return self.model