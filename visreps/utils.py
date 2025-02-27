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
from dotenv import load_dotenv

# Suppress specific torch.load FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt EXIF data.*")


def setup_logging():
    """Initialize Rich with custom theme and return themed print function"""
    custom_theme = Theme(
        {
            "info": "bold white",
            "success": "green",
            "warning": "bold yellow",
            "error": "bold red",
            "highlight": "bold magenta",
            "setup": "cyan",
        }
    )
    console = Console(theme=custom_theme)
    return console.print


rprint = setup_logging()


class ConfigVerifier:
    """Validates configuration for both training and evaluation modes."""

    VALID_MODES = {"train", "eval"}
    VALID_DATASETS = {"imagenet", "tiny-imagenet"}
    VALID_MODEL_CLASSES = {"custom_cnn", "standard_cnn"}
    VALID_MODEL_SOURCES = {"checkpoint", "torchvision"}
    VALID_REGIONS = {
        "early visual stream",
        "midventral visual stream",
        "ventral visual stream",
    }
    VALID_ANALYSES = {"rsa", "cross_decomposition"}
    VALID_NEURAL_DATASETS = {"nsd"}
    VALID_LAYERS = {"conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"}

    def __init__(self, cfg: OmegaConf):
        """Initialize verifier with configuration."""
        self.cfg = cfg
        self.rprint = setup_logging()

    def verify(self) -> OmegaConf:
        """Main verification method that routes to appropriate validator."""
        self._verify_mode()
        return self._verify_train() if self.cfg.mode == "train" else self._verify_eval()

    def _verify_mode(self) -> None:
        """Verify the configuration mode."""
        if self.cfg.mode not in self.VALID_MODES:
            self.rprint(
                f"[red]Invalid mode: {self.cfg.mode}. Must be in {self.VALID_MODES}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid mode: {self.cfg.mode}")

    def _verify_train(self) -> OmegaConf:
        """Verify training configuration."""
        self.rprint("Validating training configuration...", style="setup")

        # Dataset validation
        if self.cfg.dataset not in self.VALID_DATASETS:
            self.rprint(
                f"[red]Invalid dataset: {self.cfg.dataset}. Must be in {self.VALID_DATASETS}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid dataset: {self.cfg.dataset}")

        # Model class validation
        if self.cfg.model_class not in self.VALID_MODEL_CLASSES:
            self.rprint(
                f"[red]Invalid model_class. Must be in {self.VALID_MODEL_CLASSES}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid model_class: {self.cfg.model_class}")

        # PCA validation
        if not hasattr(self.cfg, "pca_labels"):
            self.rprint("[red]Missing required config: pca_labels[/red]", style="error")
            raise AssertionError("pca_labels flag must be specified")

        # Model-specific validations
        self._verify_model_config()

        # PCA classes validation
        if self.cfg.pca_labels:
            self._verify_pca_config()

        # Set default batch size if not specified
        if not hasattr(self.cfg, "batchsize"):
            self.cfg.batchsize = 64
            self.rprint("ℹ️  Using default batch size: 64", style="info")

        self.rprint("✅ Training configuration validation successful", style="success")
        return self.cfg

    def _verify_eval(self) -> OmegaConf:
        """Verify evaluation configuration."""
        self.rprint("Validating evaluation configuration...", style="setup")

        # Neural parameters validation
        if self.cfg.region.lower() not in self.VALID_REGIONS:
            self.rprint(
                f"[red]Invalid region: {self.cfg.region}. Must be in {self.VALID_REGIONS}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid region: {self.cfg.region}")

        if not 0 <= self.cfg.subject_idx < 8:
            self.rprint(
                f"[red]Invalid subject index: {self.cfg.subject_idx}. Must be in range [0, 7][/red]",
                style="error",
            )
            raise AssertionError(f"Invalid subject index: {self.cfg.subject_idx}")

        if self.cfg.analysis.lower() not in self.VALID_ANALYSES:
            self.rprint(
                f"[red]Invalid analysis: {self.cfg.analysis}. Must be in {self.VALID_ANALYSES}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid analysis: {self.cfg.analysis}")

        if self.cfg.neural_dataset.lower() not in self.VALID_NEURAL_DATASETS:
            self.rprint(
                "[red]Currently only NSD dataset is supported[/red]", style="error"
            )
            raise AssertionError("Currently only NSD dataset is supported")

        # Model layers validation
        if not hasattr(self.cfg.return_nodes, "__iter__"):
            self.rprint(
                f"[red]return_nodes must be a list-like object[/red]", style="error"
            )
            raise AssertionError("return_nodes must be a list-like object")

        if not self.cfg.return_nodes:
            self.rprint("[red]return_nodes list cannot be empty[/red]", style="error")
            raise AssertionError("return_nodes list cannot be empty")

        if not all(node in self.VALID_LAYERS for node in self.cfg.return_nodes):
            self.rprint(
                f"[red]Invalid return nodes: {self.cfg.return_nodes}. Must be in {self.VALID_LAYERS}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid return nodes: {self.cfg.return_nodes}")

        # Model loading validation
        if self.cfg.load_model_from not in self.VALID_MODEL_SOURCES:
            self.rprint(
                f"[red]load_model_from must be in {self.VALID_MODEL_SOURCES}[/red]",
                style="error",
            )
            raise AssertionError(
                f"load_model_from must be in {self.VALID_MODEL_SOURCES}"
            )

        if self.cfg.load_model_from == "checkpoint":
            if hasattr(self.cfg, "torchvision"):
                self.rprint(
                    "[red]torchvision key present in checkpoint mode[/red]",
                    style="error",
                )
                raise AssertionError("torchvision key not allowed in checkpoint mode")
            checkpoint_path = Path(
                f"model_checkpoints/{self.cfg.exp_name}/cfg{self.cfg.cfg_id}/{self.cfg.checkpoint_model}"
            )
            if not checkpoint_path.exists():
                self.rprint(
                    f"[red]Checkpoint not found: {checkpoint_path}[/red]", style="error"
                )
                raise AssertionError(f"Checkpoint not found: {checkpoint_path}")

        self.rprint(
            "✅ Evaluation configuration validation successful", style="success"
        )
        return self.cfg

    def _verify_model_config(self) -> None:
        """Verify model-specific configuration."""
        if self.cfg.model_class == "standard_cnn":
            if hasattr(self.cfg, "custom_cnn"):
                self.rprint(
                    "[red]Invalid config: custom_cnn key present in standard_cnn mode[/red]",
                    style="error",
                )
                raise AssertionError(
                    "custom_cnn key should not be present in standard_cnn mode"
                )
        else:  # custom_cnn
            if hasattr(self.cfg, "standard_cnn"):
                self.rprint(
                    "[red]Invalid config: standard_cnn key present in custom_cnn mode[/red]",
                    style="error",
                )
                raise AssertionError(
                    "standard_cnn key should not be present in custom_cnn mode"
                )

            # Validate custom CNN architecture parameters
            if not all(char in "01" for char in self.cfg.arch.conv_trainable):
                self.rprint(
                    "[red]Invalid conv_trainable string. Must only contain '0's and '1's[/red]",
                    style="error",
                )
                raise AssertionError("conv_trainable must only contain '0's and '1's")

            if not all(char in "01" for char in self.cfg.arch.fc_trainable):
                self.rprint(
                    "[red]Invalid fc_trainable string. Must only contain '0's and '1's[/red]",
                    style="error",
                )
                raise AssertionError("fc_trainable must only contain '0's and '1's")

            # Model-dataset compatibility warnings
            if self.cfg.dataset == "imagenet" and "tiny" in self.cfg.model_name.lower():
                self.rprint(
                    "⚠️  Training TinyCustomCNN on ImageNet-1k. This model is designed for TinyImageNet.",
                    style="warning",
                )
            elif (
                self.cfg.dataset == "tiny-imagenet"
                and "tiny" not in self.cfg.model_name.lower()
            ):
                self.rprint(
                    "⚠️  Training CustomCNN on TinyImageNet. This model is designed for ImageNet-1k.",
                    style="warning",
                )

    def _verify_pca_config(self) -> None:
        """Verify PCA-specific configuration."""
        if self.cfg.pca_n_classes <= 1:
            self.rprint(
                "[red]Invalid pca_n_classes. Must be greater than 1 when pca_labels is True[/red]",
                style="error",
            )
            raise AssertionError(
                "pca_n_classes must be greater than 1 when pca_labels is True"
            )

        if (self.cfg.pca_n_classes & (self.cfg.pca_n_classes - 1)) != 0:
            self.rprint(
                "[red]Invalid pca_n_classes. Must be a power of 2[/red]", style="error"
            )
            raise AssertionError("pca_n_classes must be a power of 2")


def validate_config(cfg: OmegaConf) -> OmegaConf:
    """Validate configuration using ConfigVerifier."""
    verifier = ConfigVerifier(cfg)
    return verifier.verify()


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

    if source_key := (cfg.load_model_from if cfg.mode == "eval" else cfg.model_class):
        other_key = {
            "eval": {"torchvision": "checkpoint", "checkpoint": "torchvision"},
            "train": {"custom_cnn": "standard_cnn", "standard_cnn": "custom_cnn"},
        }[cfg.mode][source_key]
        if other_key in cfg:
            del cfg[other_key]
        merge_nested_config(cfg, source_key)

    if cfg.mode == "eval" and cfg.load_model_from == "torchvision":
        del cfg.cfg_id

    final_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides or []))
    formatted_cfg = OmegaConf.to_yaml(final_cfg, resolve=True)
    print(f"Final Configuration:\n{formatted_cfg}\n\n")
    return final_cfg


def load_pickle(file_path):
    """Load data from pickle file"""
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(
            f"Error unpickling file at {file_path}. File may be corrupted."
        )
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
            if not key.startswith("_") and not isinstance(value, (dict, list)):
                clean_df[key] = value

    # Add random delay
    random.seed(os.urandom(10))
    time.sleep(random.uniform(0, 3))

    # Setup paths and lock
    save_dir = Path("logs") / cfg.mode / cfg.load_model_from
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
        rprint(
            f"ERROR: Could not acquire lock for {results_path} after {timeout}s",
            style="error",
        )
        raise
    except Exception as e:
        rprint(
            f"ERROR: Failed to save results to {results_path}: {str(e)}", style="error"
        )
        raise

    return str(results_path)


def get_optimizer_class(optimizer_name):
    """Get optimizer class by name with exact or fuzzy matching."""
    available_optimizers = {
        name.lower(): getattr(optim, name)
        for name in dir(optim)
        if name[0].isupper() and not name.startswith("_")
    }

    opt_name = optimizer_name.lower()
    if opt_name in available_optimizers:
        return available_optimizers[opt_name]

    matches = [
        name
        for name in available_optimizers.keys()
        if name.startswith(opt_name) or opt_name.startswith(name)
    ]
    if matches:
        return available_optimizers[matches[0]]

    raise ValueError(
        f"Could not find optimizer '{optimizer_name}'. Available optimizers: {list(available_optimizers.keys())}"
    )


def calculate_cls_accuracy(data_loader, model, device):
    """Calculate classification accuracies with proper device handling and numerical stability.

    For models with fewer than 5 classes, only top-1 accuracy is computed (top-5 is returned as an empty string).
    For models with 5 or more classes, both top-1 and top-5 accuracies are computed.

    Args:
        data_loader: PyTorch DataLoader
        model: PyTorch model
        device: torch.device for computation

    Returns:
        tuple: (top1_accuracy, top5_accuracy) as percentages (0-100). For small number of classes, top5_accuracy = ""
    """
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0

    # Choose autocast settings based on device type
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    # This flag will be set based on the model's output dimension (only once)
    use_top5 = None

    with (
        torch.no_grad(),
        torch.autocast(device_type=autocast_device, dtype=autocast_dtype),
    ):
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            batch_size = labels.size(0)
            total += batch_size

            # Determine if we should compute top-5 accuracy (if there are at least 5 classes)
            if use_top5 is None:
                use_top5 = outputs.size(1) >= 5

            if not use_top5:
                # Only top-1 is relevant
                _, preds = outputs.max(dim=1)
                top1_correct += (preds == labels).sum().item()
            else:
                # Compute top-1 and top-5
                maxk = 5
                _, preds = outputs.topk(maxk, dim=1, largest=True, sorted=True)
                preds = preds.t()  # shape: [maxk, batch_size]
                correct = preds.eq(labels.view(1, -1).expand_as(preds))
                top1_correct += correct[0].sum().item()
                # For top-5, an instance is correct if any of the top-5 predictions match the label
                top5_correct += correct.sum(dim=0).gt(0).sum().item()

    if total == 0:
        return 0.0, 0.0

    top1_acc = 100.0 * top1_correct / total
    if not use_top5:
        return top1_acc, ""
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc


class Logger:
    """Unified logging class for training metrics and system stats."""

    def __init__(self, use_wandb=True, log_system_metrics=True, cfg=None):
        self.use_wandb = use_wandb
        self.log_system_metrics = log_system_metrics

        if use_wandb:
            try:
                import wandb

                self.wandb = wandb

                # Check if wandb is logged in
                if not wandb.api.api_key:
                    rprint(
                        "WandB not authenticated. Please run 'wandb login' first.",
                        style="error",
                    )
                    self.use_wandb = False
                    return

                # Set environment variables for minimal output
                os.environ["WANDB_SILENT"] = "true"

                # Initialize wandb if config provided
                if cfg is not None:
                    group = f"seed_{cfg.seed}"
                    name = f"{cfg.model_name}_{cfg.model_class}"
                    tags = [cfg.model_class, f"lr_{cfg.learning_rate}"]

                    self.wandb.init(
                        entity="visreps",  # Use team name
                        project=cfg.dataset,
                        group=group,
                        name=name,
                        config=OmegaConf.to_container(cfg, resolve=True),
                        tags=tags,
                        notes=f"Training {cfg.model_name} with seed {cfg.seed}",
                        settings=wandb.Settings(start_method="thread"),
                    )
                    rprint(
                        f"WandB initialized. View results at: {wandb.run.get_url()}",
                        style="info",
                    )

                    # Use epoch as x-axis for all metrics
                    wandb.define_metric("*", step_metric="epoch")

            except (ImportError, Exception) as e:
                rprint(
                    f"W&B import failed with error: {str(e)}\nFull error: {repr(e)}",
                    style="error",
                )
                self.use_wandb = False

    def log(self, metrics):
        """Log metrics to wandb."""
        if self.use_wandb:
            try:
                self.wandb.log(metrics)
            except Exception as e:
                rprint(f"W&B logging failed: {str(e)}", style="warning")


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
    """Setup optimizer with proper weight decay and parameters."""
    # Separate parameters that should and shouldn't use weight decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't use weight decay on biases and batch norm parameters
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    parameters = [
        {"params": decay, "weight_decay": cfg.get("weight_decay", 0.0)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # Setup optimizer
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(parameters, lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def setup_scheduler(optimizer, cfg):
    """Setup learning rate scheduler with warmup."""
    # First set up the warmup
    warmup_epochs = cfg.get("warmup_epochs", 0)

    if warmup_epochs > 0:
        # Create warmup scheduler that linearly increases LR from 10% to 100% of base LR
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of base LR
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Main cosine scheduler that goes from 100% to 1% of base LR
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs - warmup_epochs,  # Remaining epochs after warmup
            eta_min=cfg.learning_rate * 0.05,  # Minimum LR is 5% of initial LR
        )

        # Combine schedulers
        return torch.optim.lr_scheduler.ChainedScheduler([warmup, main])
    else:
        # If no warmup, just use cosine annealing
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.05
        )


def log_training_step(logger, cfg, epoch, batch_idx, loss, lr):
    """Log training step metrics."""
    if cfg.use_wandb:
        # Convert batch_idx to fractional epoch for smooth curves
        fractional_epoch = (
            epoch - 1 + (batch_idx / logger.wandb.run.config.train_loader_len)
        )
        logger.log(
            {
                "epoch": fractional_epoch,
                "training/loss": loss,
                "training/learning_rate": lr,
            }
        )


def log_training_metrics(logger, cfg, epoch, loss, metrics, scheduler):
    """Log training metrics with rich console output."""
    if cfg.use_wandb:
        # Log all metrics under training namespace
        log_dict = {"epoch": epoch, "training/test-acc": metrics["test_acc"]}

        # Add train accuracy if available
        if "train_acc" in metrics:
            log_dict["training/train-acc"] = metrics["train_acc"]

        # Add top5 metrics if available and not using PCA labels
        if not cfg.pca_labels:
            if "test_top5" in metrics:
                log_dict["training/test-top5"] = metrics["test_top5"]
            if "train_top5" in metrics:
                log_dict["training/train-top5"] = metrics["train_top5"]

        logger.log(log_dict)

    # Print metrics every epoch
    if cfg.pca_labels:
        status = (
            f"Epoch [{epoch}/{cfg.num_epochs}] Test Acc: {metrics['test_acc']:.2f}%"
        )
        if "train_acc" in metrics:
            status += f" Train Acc: {metrics['train_acc']:.2f}%"
    else:
        status = (
            f"Epoch [{epoch}/{cfg.num_epochs}] Test Acc: {metrics['test_acc']:.2f}%"
        )
        if "test_top5" in metrics:
            status += f" (top5: {metrics['test_top5']:.2f}%)"
        if "train_acc" in metrics:
            status += f" Train Acc: {metrics['train_acc']:.2f}%"
            if "train_top5" in metrics:
                status += f" (top5: {metrics['train_top5']:.2f}%)"
    rprint(status, style="info")


def get_env_var(key):
    """Get path from environment variable or raise error if not found"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    path = os.environ.get(key)
    if path is None:
        raise ValueError(
            f"Environment variable {key} not found. Please set it in .env file or system environment."
        )
    return path
