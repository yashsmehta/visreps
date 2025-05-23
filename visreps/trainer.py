import torch
import torch.nn as nn
import time
from tqdm import tqdm
from collections import defaultdict

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models import utils as model_utils
import visreps.utils as utils
from visreps.utils import calculate_cls_accuracy, is_interactive_environment, MetricsLogger


class Trainer:
    """Trainer class for object classification models."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = None
        self.setup_environment()
        self.setup_training()

    def setup_environment(self):
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_training(self):
        self.datasets, self.loaders = get_obj_cls_loader(self.cfg)
        num_classes = self.cfg.pca_n_classes if self.cfg.pca_labels else self.datasets["train"].num_classes
        self.model = model_utils.load_model(self.cfg, self.device, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = utils.setup_optimizer(self.model, self.cfg)
        self.scheduler = utils.setup_scheduler(self.optimizer, self.cfg)
        
        # Setup metrics logger and checkpoint directory
        self.checkpoint_dir = None
        if self.cfg.log_checkpoints:
            self.checkpoint_dir, self.cfg_dict = model_utils.setup_checkpoint_dir(self.cfg, self.model)
            model_utils.save_checkpoint(self.checkpoint_dir, 0, self.model, self.optimizer, {}, self.cfg_dict)
        
        if self.cfg.use_wandb:
            self.cfg.train_loader_len = len(self.loaders["train"])
        
        self.metrics_logger = MetricsLogger(self.cfg, self.checkpoint_dir)
        self._num_classes = num_classes

    @torch.no_grad()
    def evaluate(self, split="test"):
        self.model.eval()
        return calculate_cls_accuracy(self.loaders[split], self.model, self.device)

    def train_epoch(self, epoch):
        self.model.train()
        
        # Only use tqdm progress bar in interactive environments
        train_loader = self.loaders["train"]
        if is_interactive_environment():
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        else:
            print(f"Starting Epoch {epoch}")
            
        epoch_stats = defaultdict(float)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            epoch_stats['batch_loss'] += loss.item()
            epoch_stats['n_batches'] = i + 1

            loss.backward()
            if hasattr(self.cfg, 'grad_clip') and self.cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            else:
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None]))
            epoch_stats['grad_norm'] += grad_norm.item()
            self.optimizer.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            epoch_stats['learning_rate'] = curr_lr

            # Log training step metrics
            self.metrics_logger.log_training_step(epoch, i, loss.item(), curr_lr)

            avg_loss = epoch_stats['batch_loss'] / (i + 1)
            # Only update progress bar in interactive environments
            if is_interactive_environment():
                train_loader.set_postfix({
                    'Avg Loss': f'{avg_loss:.4f}',
                    'LR': f'{curr_lr:.6f}',
                    'Grad Norm': f'{grad_norm:.4f}'
                })
            # Print occasional updates in non-interactive environments
            elif i % 10000 == 0:
                print(f"Batch {i}/{len(self.loaders['train'])}, Avg Loss: {avg_loss:.4f}, LR: {curr_lr:.6f}")

        n_batches = epoch_stats['n_batches']
        avg_epoch_loss = epoch_stats['batch_loss'] / n_batches
        epoch_metrics = {
            'epoch_loss': avg_epoch_loss,
            'learning_rate': curr_lr
        }
        return avg_epoch_loss, epoch_metrics

    def train(self):
        for epoch in range(1, self.cfg.num_epochs + 1):
            epoch_start_time = time.time()
            epoch_loss, epoch_metrics = self.train_epoch(epoch)
            self.scheduler.step()

            metrics = {
                "epoch": epoch,
                "epoch_metrics": epoch_metrics
            }

            if epoch % self.cfg.log_interval == 0:
                test_top1, test_top5 = self.evaluate("test")
                train_top1, train_top5 = self.evaluate("train")

                metrics.update({
                    "test_acc": test_top1,
                    "test_top5": test_top5,
                    "train_acc": train_top1,
                    "train_top5": train_top5,
                })
                
                self.metrics_logger.log_metrics(epoch, epoch_loss, metrics)
                
            if self.cfg.log_checkpoints and epoch % self.cfg.checkpoint_interval == 0:
                model_utils.save_checkpoint(
                    self.checkpoint_dir, epoch, self.model, self.optimizer, metrics, self.cfg_dict
                )
        
        self.metrics_logger.finish()
        return self.model