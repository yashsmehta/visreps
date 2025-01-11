import time
import random
from pathlib import Path
import os
import pickle
import warnings
import torch
import torch.optim as optim
from filelock import FileLock, Timeout
from rich.console import Console
from rich.theme import Theme
from omegaconf import OmegaConf

# Suppress specific torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="You are using `torch.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", category=UserWarning,
                       message="Corrupt EXIF data.*")

def setup_logging():
    """Initialize Rich with custom theme and return themed print function"""
    custom_theme = Theme({
        "info": "bold white",
        "success": "green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "setup": "cyan"
    })
    console = Console(theme=custom_theme)
    return console.print

# Initialize Rich print globally as rprint
rprint = setup_logging()

def check_trainer_config(cfg):
    """
    Validates the trainer configuration for the number of elements and content in 'conv_trainable' and 'fc_trainable'.
    Also validates and sets dataset-specific parameters like num_classes and default batch size.

    Args:
        cfg (OmegaConf): The configuration object containing training parameters.

    Returns:
        OmegaConf: The validated and potentially modified configuration object.

    Raises:
        AssertionError: If any of the configuration conditions are not met.
    """
    # Check model class
    assert cfg.model_class in [
        "custom_cnn",
        "standard_cnn",
    ], "model_class must be one of 'custom_cnn', 'standard_cnn'!"
    
    # Set dataset-specific num_classes
    if cfg.dataset == "imagenet":
        cfg.num_classes = 1000
    elif cfg.dataset == "tiny-imagenet":
        cfg.num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    
    # Set default batch size if not specified
    if not hasattr(cfg, "batchsize"):
        cfg.batchsize = 32
        rprint(f"Using default batch size: {cfg.batchsize}", style="info")
    
    # Check custom CNN architecture parameters if applicable
    if cfg.model_class == "custom_cnn":
        assert all(
            char in "01" for char in cfg.arch.conv_trainable
        ), "conv_trainable must only contain '0's and '1's!"
        assert all(
            char in "01" for char in cfg.arch.fc_trainable
        ), "fc_trainable must only contain '0's and '1's!"
    
    return cfg


def merge_nested_config(cfg, source_key):
    """Merge nested config into root."""
    if source_key not in cfg:
        return
    
    source = OmegaConf.to_container(cfg[source_key], resolve=True)
    cfg.update(source)
    del cfg[source_key]


def load_config(config_path, overrides):
    """Load config from file and apply CLI overrides."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    
    if cfg.mode == "train":
        assert cfg.model_class in ["custom_cnn", "standard_cnn"], "model_class must be custom_cnn or standard_cnn"
    else:
        assert cfg.load_model_from in ["checkpoint", "torchvision"], "load_model_from must be checkpoint or torchvision"
    
    if source_key := (cfg.load_model_from if cfg.mode == "eval" else cfg.model_class):
        other_key = {"eval": {"torchvision": "checkpoint", "checkpoint": "torchvision"},
                    "train": {"custom_cnn": "standard_cnn", "standard_cnn": "custom_cnn"}}[cfg.mode][source_key]
        if other_key in cfg:
            del cfg[other_key]
        merge_nested_config(cfg, source_key)

    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides or []))


def load_pickle(file_path):
    """Load data from pickle file"""
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(f"Error unpickling file at {file_path}. File may be corrupted.")
    except Exception as e:
        raise RuntimeError(f"Error loading pickle file at {file_path}: {str(e)}")


def save_results(df, cfg, timeout=60):
    """Save results to CSV with file locking in logs/mode/exp_name.csv format.
    Adds all config parameters from OmegaConf while avoiding metadata."""
    # Create a clean DataFrame without the metadata
    clean_df = df.copy()
    
    # Convert OmegaConf to primitive container and add all config params
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if not key.startswith('_') and not isinstance(value, (dict, list)):
                clean_df[key] = value

    # Add random delay
    random.seed(os.urandom(10))
    time.sleep(random.uniform(0, 5))
    
    # Setup paths and lock
    save_dir = Path('logs') / cfg.mode / cfg.load_model_from
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"{cfg.exp_name}.csv"
    lock_path = results_path.with_suffix(".lock")
    lock = FileLock(str(lock_path), timeout=timeout)
    
    try:
        with lock:
            write_header = not results_path.exists()
            clean_df.to_csv(results_path, mode="a", header=write_header, index=False)
            rprint(f"Successfully saved results to {results_path}", style="success")
        # Remove lock file after successful save
        if lock_path.exists():
            lock_path.unlink()
    except Timeout:
        rprint(f"ERROR: Could not acquire lock for {results_path} after {timeout}s", style="error")
        raise
    except Exception as e:
        rprint(f"ERROR: Failed to save results to {results_path}: {str(e)}", style="error")
        raise
        
    return str(results_path)


def get_optimizer_class(optimizer_name):
    """Get optimizer class by name with exact or fuzzy matching."""
    available_optimizers = {name.lower(): getattr(optim, name) 
                          for name in dir(optim) 
                          if name[0].isupper() and not name.startswith('_')}
    
    opt_name = optimizer_name.lower()
    if opt_name in available_optimizers:
        return available_optimizers[opt_name]
    
    matches = [name for name in available_optimizers.keys() 
              if name.startswith(opt_name) or opt_name.startswith(name)]
    if matches:
        return available_optimizers[matches[0]]
    
    raise ValueError(f"Could not find optimizer '{optimizer_name}'. Available optimizers: {list(available_optimizers.keys())}")


def calculate_cls_accuracy(data_loader, model, device):
    """Calculate classification accuracy with proper device handling and numerical stability.
    
    Args:
        data_loader: PyTorch DataLoader
        model: PyTorch model
        device: torch.device for computation
    
    Returns:
        float: Classification accuracy as percentage (0-100)
    """
    model.eval()  # Ensure model is in eval mode
    correct = 0
    total = 0
    
    # Use autocast based on device type
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        for images, labels in data_loader:
            # Move data to device efficiently
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            
            # Accumulate statistics without keeping autograd history
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()
    
    # Handle edge case and ensure floating point division
    if total == 0:
        return 0.0
        
    # Use float for stable division and percentage calculation
    return (100.0 * correct) / total

class Logger:
    """Unified logging class for training metrics and system stats."""
    def __init__(self, use_wandb=True, log_system_metrics=True, cfg=None):
        self.use_wandb = use_wandb
        self.log_system_metrics = log_system_metrics
        self.global_step = 0  # Add global step counter
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                # Check if wandb is logged in
                if not wandb.api.api_key:
                    rprint("WandB not authenticated. Please run 'wandb login' first.", style="error")
                    self.use_wandb = False
                    return
                
                # Set environment variables for minimal output
                os.environ['WANDB_SILENT'] = 'true'
                
                # Initialize wandb if config provided
                if cfg is not None:
                    project = "imagenet-train"  # Use specific project name
                    group = getattr(cfg, 'group', None)
                    name = f"seed_{cfg.seed}"
                    tags = [cfg.model_name, f"lr_{cfg.learning_rate}"]
                    
                    self.wandb.init(
                        entity="visreps",  # Use team name
                        project=project,
                        group=group,
                        name=name,
                        config=OmegaConf.to_container(cfg, resolve=True),
                        tags=tags,
                        notes=f"Training {cfg.model_name} with seed {cfg.seed}",
                        settings=wandb.Settings(start_method="thread")
                    )
                    rprint(f"WandB initialized. View results at: {wandb.run.get_url()}", style="info")
                
                    # Use epoch as x-axis for all metrics
                    wandb.define_metric("*", step_metric="epoch")
                
            except (ImportError, Exception) as e:
                rprint(f"W&B import failed with error: {str(e)}\nFull error: {repr(e)}", style="error")
                self.use_wandb = False
        
    def _get_system_metrics(self):
        """Collect system metrics if enabled."""
        if not self.log_system_metrics:
            return {}
            
        try:
            import psutil
            import GPUtil
            
            metrics = {
                'system/memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'system/cpu_percent': psutil.cpu_percent()
            }
            
            if torch.cuda.is_available():
                metrics.update({
                    'system/gpu_utilization': GPUtil.getGPUs()[0].load,
                    'system/gpu_memory': torch.cuda.memory_allocated() / 1024 / 1024
                })
                
            return metrics
        except Exception:
            return {}  # Fail silently on system metrics errors
        
    def log_batch(self, batch_idx, metrics):
        """Log batch-level metrics with optional system stats."""
        if not self.use_wandb:
            return
            
        try:
            combined_metrics = {
                "global_step": self.global_step,
                **metrics,
                **self._get_system_metrics()
            }
            self.wandb.log(combined_metrics)
            self.global_step += 1  # Increment global step
        except Exception as e:
            rprint(f"W&B batch logging failed: {str(e)}", style="warning")
        
    def log_epoch(self, metrics):
        """Log epoch-level metrics with optional system stats."""
        if not self.use_wandb:
            return
            
        try:
            combined_metrics = {
                "global_step": self.global_step,
                **metrics,
                **self._get_system_metrics()
            }
            self.wandb.log(combined_metrics)
        except Exception as e:
            rprint(f"W&B epoch logging failed: {str(e)}", style="warning")
        
    def log_gradients(self, model, epoch):
        """Log model gradient histograms."""
        if not self.use_wandb:
            return
            
        try:
            gradient_dict = {
                "global_step": self.global_step,
                **{f'gradients/{name}': self.wandb.Histogram(param.grad.cpu().numpy())
                   for name, param in model.named_parameters()
                   if param.grad is not None}
            }
            self.wandb.log(gradient_dict)
        except Exception as e:
            rprint(f"W&B gradient logging failed: {str(e)}", style="warning")

# Global logger instance
_logger = None

def get_logger(use_wandb=True, log_system_metrics=True, cfg=None):
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger(use_wandb, log_system_metrics, cfg)
    return _logger

def log_batch_metrics(batch_idx, metrics, use_wandb=True):
    """Log batch-level training metrics to W&B."""
    get_logger(use_wandb).log_batch(batch_idx, metrics)

def log_epoch_metrics(epoch_metrics, use_wandb=True):
    """Log epoch-level training metrics to W&B."""
    get_logger(use_wandb).log_epoch(epoch_metrics)

def setup_optimizer(model, cfg):
    """Setup optimizer with weight decay."""
    optimizer_cls = get_optimizer_class(cfg.optimizer)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0}
    ]
    return optimizer_cls(param_groups, lr=cfg.learning_rate)

def setup_scheduler(optimizer, cfg, steps_per_epoch):
    """Setup learning rate scheduler with warmup."""
    warmup_epochs = 5
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_epochs/cfg.num_epochs
    )

def log_training_step(logger, cfg, batch_idx, loss, lr):
    """Log training step metrics."""
    # No batch-level logging - progress bar will handle this
    pass

def log_training_metrics(logger, cfg, epoch, loss, metrics, scheduler):
    """Log training metrics with rich console output."""
    if cfg.use_wandb:
        # Log all metrics under training namespace
        log_dict = {
            'epoch': epoch,
            'training/loss': loss,
            'training/test-acc': metrics['test_acc']
        }
        
        # Add train accuracy if enabled
        if 'train_acc' in metrics:
            log_dict['training/train-acc'] = metrics['train_acc']
            
        logger.wandb.log(log_dict)
        
    # Print metrics every epoch
    status = f"Epoch [{epoch}/{cfg.num_epochs}] Loss: {loss:.6f} Test Acc: {metrics['test_acc']:.2f}%"
    if 'train_acc' in metrics:
        status += f" Train Acc: {metrics['train_acc']:.2f}%"
    rprint(status, style="info")