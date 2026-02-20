import sqlite3
import hashlib
import json
from pathlib import Path
import os
import pickle
import warnings
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, LinearLR, SequentialLR
from rich.console import Console
from rich.theme import Theme
from omegaconf import OmegaConf
from dotenv import load_dotenv
import sys
import csv
import wandb
import pandas as pd

# print("--- Importing visreps/utils.py ---")

# Suppress specific torch.load FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt EXIF data.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter.*scheduler.step.*")


def is_interactive_environment():
    """
    Detect if code is running in an interactive environment (like a terminal or Jupyter notebook)
    rather than in a batch job (like SLURM).
    
    Returns:
        bool: True if running in an interactive environment, False otherwise
    """
    # Check if running in a SLURM job
    if os.environ.get('SLURM_JOB_ID') is not None:
        return False
    
    # Check if running in a Jupyter notebook
    try:
        if 'ipykernel' in sys.modules:
            return True
    except:
        pass
    
    # Check if stdout is attached to a terminal
    try:
        return sys.stdout.isatty()
    except:
        return False


def setup_logging():
    """Initialize Rich with custom theme and return (console, print) tuple."""
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
    return console, console.print


console, rprint = setup_logging()


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
                # Ensure labels are on the same device as predictions for comparison
                correct = preds.eq(labels.to(preds.device).view(1, -1).expand_as(preds)) 
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


class MetricsLogger:
    """Unified class for logging metrics to both CSV and wandb."""
    def __init__(self, cfg, checkpoint_dir=None):
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.metrics_file = None
        
        # Setup CSV logging
        if checkpoint_dir:
            self.metrics_file = os.path.join(checkpoint_dir, 'training_metrics.csv')
            metrics_fieldnames = ['epoch', 'train_loss', 'train_acc', 'train_top5', 
                                'test_acc', 'test_top5', 'learning_rate']
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
                writer.writeheader()
        
        # Setup wandb logging
        self.use_wandb = cfg.use_wandb
        if self.use_wandb:
            try:
                # Check if wandb is logged in
                if not wandb.api.api_key:
                    rprint("WandB not authenticated. Please run 'wandb login' first.", style="error")
                    self.use_wandb = False
                    return

                # Set environment variables for minimal output
                os.environ["WANDB_SILENT"] = "true"

                # Initialize wandb
                group = f"seed_{cfg.seed}"
                name = f"{cfg.model_name}_{cfg.model_class}"
                tags = [cfg.model_class, f"lr_{cfg.learning_rate}"]

                wandb.init(
                    entity="visreps",
                    project=cfg.dataset,
                    group=group,
                    name=name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    tags=tags,
                    notes=f"Training {cfg.model_name} with seed {cfg.seed}",
                    settings=wandb.Settings(start_method="thread"),
                )
                
                # Use epoch as x-axis for all metrics
                wandb.define_metric("*", step_metric="epoch")
                rprint(f"WandB initialized. View results at: {wandb.run.get_url()}", style="info")

            except Exception as e:
                rprint(f"W&B initialization failed: {str(e)}", style="error")
                self.use_wandb = False

    def log_metrics(self, epoch, loss, metrics):
        """Log metrics to both CSV and wandb if configured."""
        # Log to CSV
        if self.metrics_file:
            csv_metrics = {
                'epoch': metrics['epoch'],
                'train_loss': loss,
                'train_acc': metrics.get('train_acc', ''),
                'train_top5': metrics.get('train_top5', ''),
                'test_acc': metrics.get('test_acc', ''),
                'test_top5': metrics.get('test_top5', ''),
                'learning_rate': metrics['epoch_metrics']['learning_rate']
            }
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_metrics.keys())
                writer.writerow(csv_metrics)

        # Log to wandb
        if self.use_wandb:
            log_dict = {'epoch': epoch, 'training/test-acc': metrics['test_acc']}
            
            # Add train accuracy if available
            if 'train_acc' in metrics:
                log_dict['training/train-acc'] = metrics['train_acc']
            
            # Add top5 metrics if available and not using PCA labels
            if not self.cfg.pca_labels:
                if 'test_top5' in metrics:
                    log_dict['training/test-top5'] = metrics['test_top5']
                if 'train_top5' in metrics:
                    log_dict['training/train-top5'] = metrics['train_top5']
            
            try:
                wandb.log(log_dict)
            except Exception as e:
                rprint(f"W&B logging failed: {str(e)}", style="warning")

        # Print metrics
        if self.cfg.pca_labels:
            status = f"Epoch [{epoch}/{self.cfg.num_epochs}] Test Acc: {metrics['test_acc']:.2f}%"
            if 'train_acc' in metrics:
                status += f" Train Acc: {metrics['train_acc']:.2f}%"
        else:
            status = f"Epoch [{epoch}/{self.cfg.num_epochs}] Test Acc: {metrics['test_acc']:.2f}%"
            if 'test_top5' in metrics:
                status += f" (top5: {metrics['test_top5']:.2f}%)"
            if 'train_acc' in metrics:
                status += f" Train Acc: {metrics['train_acc']:.2f}%"
                if 'train_top5' in metrics:
                    status += f" (top5: {metrics['train_top5']:.2f}%)"
        rprint(status, style="info")

    def log_training_step(self, epoch, batch_idx, loss, lr):
        """Log training step metrics to wandb."""
        if self.use_wandb:
            try:
                fractional_epoch = epoch - 1 + (batch_idx / self.cfg.train_loader_len)
                wandb.log({
                    'epoch': fractional_epoch,
                    'training/loss': loss,
                    'training/learning_rate': lr
                })
            except Exception as e:
                rprint(f"W&B step logging failed: {str(e)}", style="warning")

    def finish(self):
        """Cleanup and finish logging."""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                rprint(f"W&B finish failed: {str(e)}", style="warning")


def get_env_var(key):
    """Get path from environment variable. Attempts to load .env if key not found initially."""
    # print(f"--- Inside visreps.utils.get_env_var for key: {key} ---")
    load_dotenv()
    path = os.environ.get(key)
    if path is None:
        # If not found, print debug info and return an empty string to avoid TypeError downstream.
        # This assumes downstream code can handle an empty path or will raise its own error.
        print(f"Debug: Environment variable '{key}' not found in os.environ.")
        print(f"Debug: Current os.environ keys (first 50): {list(os.environ.keys())[:50]}") 
        return "" # Return empty string instead of raising error or returning None
    return path


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


_RESULTS_DB_PATH = Path("results.db")

_IDENTITY_FIELDS = (
    "seed", "epoch", "region", "subject_idx", "neural_dataset", "cfg_id",
    "pca_labels", "pca_n_classes", "pca_labels_folder", "checkpoint_dir",
    "analysis", "compare_method", "reconstruct_from_pcs", "pca_k",
)


def _compute_run_id(cfg) -> str:
    """Deterministic hash of experiment identity fields."""
    identity = {f: cfg.get(f) for f in _IDENTITY_FIELDS}
    identity["subject_idx"] = str(identity.get("subject_idx"))
    raw = json.dumps(identity, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _init_db(db_path) -> sqlite3.Connection:
    """Open (or create) the results SQLite database with WAL mode."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            run_id              TEXT NOT NULL,
            compare_method      TEXT NOT NULL,
            layer               TEXT NOT NULL,
            score               REAL,
            ci_low              REAL,
            ci_high             REAL,
            analysis            TEXT NOT NULL,
            seed                INTEGER NOT NULL,
            epoch               INTEGER NOT NULL,
            region              TEXT,
            subject_idx         TEXT,
            neural_dataset      TEXT NOT NULL,
            cfg_id              INTEGER,
            pca_labels          BOOLEAN NOT NULL,
            pca_n_classes       INTEGER,
            pca_labels_folder   TEXT,
            model_name          TEXT NOT NULL,
            checkpoint_dir      TEXT,
            reconstruct_from_pcs BOOLEAN DEFAULT 0,
            pca_k               INTEGER DEFAULT 1,
            UNIQUE(run_id, compare_method, layer)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_configs (
            run_id      TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS layer_selection_scores (
            run_id          TEXT NOT NULL,
            compare_method  TEXT NOT NULL,
            layer           TEXT NOT NULL,
            score           REAL,
            UNIQUE(run_id, compare_method, layer)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bootstrap_distributions (
            run_id          TEXT NOT NULL,
            compare_method  TEXT NOT NULL,
            scores          TEXT,
            UNIQUE(run_id, compare_method)
        )
    """)
    conn.commit()
    return conn


def _get_float(row, col):
    """Safely extract a float from a DataFrame row, returning None if missing/NaN."""
    if col in row.index and pd.notna(row.get(col)):
        return float(row[col])
    return None


def save_results(df, cfg, timeout=60):
    """Save evaluation results to SQLite database at results.db.

    Uses a normalized "long" format: each comparison metric (Spearman, Kendall)
    gets its own row, distinguished by the `compare_method` column. This applies
    to all tables: `results`, `layer_selection_scores`, and
    `bootstrap_distributions`. Re-running the same eval replaces old rows.
    """
    run_id = _compute_run_id(cfg)
    conn = _init_db(_RESULTS_DB_PATH)

    config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    conn.execute(
        "INSERT OR REPLACE INTO run_configs (run_id, config_json) VALUES (?, ?)",
        (run_id, config_json),
    )

    # ── results ──────────────────────────────────────────────
    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        layer = row.get("layer")
        score = _get_float(row, "score")
        ci_low = _get_float(row, "ci_low")
        ci_high = _get_float(row, "ci_high")
        if score is None:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO results
               (run_id, compare_method, layer, score, ci_low, ci_high,
                analysis, seed, epoch, region, subject_idx,
                neural_dataset, cfg_id, pca_labels, pca_n_classes, pca_labels_folder,
                model_name, checkpoint_dir, reconstruct_from_pcs, pca_k)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, method, layer, score, ci_low, ci_high,
                row.get("analysis", cfg.get("analysis")),
                int(cfg.get("seed")),
                int(cfg.get("epoch", 0)),
                cfg.get("region"),
                str(cfg.get("subject_idx")),
                cfg.get("neural_dataset"),
                cfg.get("cfg_id"),
                bool(cfg.get("pca_labels")),
                cfg.get("pca_n_classes"),
                cfg.get("pca_labels_folder"),
                cfg.get("model_name"),
                cfg.get("checkpoint_dir"),
                bool(cfg.get("reconstruct_from_pcs", False)),
                cfg.get("pca_k", 1),
            ),
        )

    # ── layer_selection_scores ────────────────────────────────
    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        entries = row.get("layer_selection_scores") or []
        for entry in entries:
            conn.execute(
                """INSERT OR REPLACE INTO layer_selection_scores
                   (run_id, compare_method, layer, score) VALUES (?, ?, ?, ?)""",
                (run_id, method, entry["layer"], float(entry["score"])),
            )

    # ── bootstrap_distributions ───────────────────────────────
    for _, row in df.iterrows():
        method = row.get("compare_method", cfg.get("compare_method", "spearman"))
        bs = row.get("bootstrap_scores")
        if bs is not None:
            conn.execute(
                """INSERT OR REPLACE INTO bootstrap_distributions
                   (run_id, compare_method, scores) VALUES (?, ?, ?)""",
                (run_id, method, json.dumps(bs)),
            )

    conn.commit()
    conn.close()
    rprint(f"Saved {len(df)} results to {_RESULTS_DB_PATH} (run_id={run_id})", style="success")
    return str(_RESULTS_DB_PATH)


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


def load_config(config_path, overrides=None):
    """Load config from file and apply CLI overrides."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)

    # First pass: Apply overrides to determine which nested config to use
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Merge the appropriate nested config based on (potentially overridden) mode and model_class/load_model_from
    if source_key := (cfg.load_model_from if cfg.mode == "eval" else cfg.model_class):
        other_key = {
            "eval": {"torchvision": "checkpoint", "checkpoint": "torchvision"},
            "train": {"custom_model": "standard_model", "standard_model": "custom_model"},
        }[cfg.mode][source_key]
        if other_key in cfg:
            del cfg[other_key]
        merge_nested_config(cfg, source_key)

    # Second pass: Apply overrides again to ensure they take final precedence over nested config
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    if cfg.mode == "eval" and cfg.load_model_from == "torchvision":
        del cfg.cfg_id

    if cfg.get("verbose", False):
        formatted_cfg = OmegaConf.to_yaml(cfg, resolve=True)
        rprint(f"Final Configuration:\n{formatted_cfg}\n")
    return cfg


class ConfigVerifier:
    """Validates configuration for both training and evaluation modes."""

    VALID_MODES = {"train", "eval"}
    VALID_DATASETS = {"imagenet", "tiny-imagenet", "imagenet-mini-10", "imagenet-mini-50", "imagenet-mini-200"}
    VALID_MODEL_CLASSES = {"custom_model", "standard_model"}
    VALID_MODEL_SOURCES = {"checkpoint", "torchvision"}
    VALID_ANALYSES = {"rsa", "encoding_score"}
    VALID_COMPARE_METHODS = {"spearman", "kendall"}
    VALID_NEURAL_DATASETS = {"nsd", "things-behavior", "tvsd"}
    VALID_NSD_TYPES = {"finegrained", "streams", "streams_shared"}

    def __init__(self, cfg: OmegaConf):
        """Initialize verifier with configuration."""
        self.cfg = cfg
        self.rprint = rprint

    def verify(self) -> OmegaConf:
        """Main verification method that routes to appropriate validator."""
        self._verify_mode()
        if self.cfg.mode == "train":
            return self._verify_train()
        else:
            return self._verify_eval()

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

        if self.cfg.get("verbose", False):
            self.rprint("✅ Config validated", style="success")
        return self.cfg

    def _verify_eval(self) -> OmegaConf:
        """Verify evaluation configuration.

        Accepts list-valued subject_idx and region for NSD/TVSD (multi-subject
        evaluation in a single process). Normalizes scalars to lists.
        """
        from omegaconf import ListConfig

        # Seed validation: ensure seed is one of [1, 2, 3]
        if getattr(self.cfg, "seed", None) not in (1, 2, 3):
            self.rprint(
                f"[red]Invalid seed: {self.cfg.seed}. Must be one of [1, 2, 3][/red]",
                style="error",
            )
            raise AssertionError(f"Invalid seed: {self.cfg.seed}")

        # Neural parameters validation
        if self.cfg.neural_dataset.lower() == "things-behavior":
            # For 'things-behavior' dataset, region and subject_idx are not applicable.
            # Set to "N/A" if they are not already "N/A".
            region = self.cfg.get("region")
            if region is not None and not (isinstance(region, str) and region.upper() == "N/A"):
                self.rprint(
                    f"[warning]Region '{region}' provided for 'things-behavior' dataset. Setting to 'N/A'.[/warning]",
                    style="warning",
                )
                self.cfg.region = "N/A"

            subj = self.cfg.get("subject_idx")
            if subj is not None and not (isinstance(subj, str) and subj.upper() == "N/A"):
                self.rprint(
                    f"[warning]Subject index '{subj}' provided for 'things-behavior' dataset. Setting to 'N/A'.[/warning]",
                    style="warning",
                )
                self.cfg.subject_idx = "N/A"

        # Remove nsd_type for non-NSD datasets (only meaningful for NSD)
        if self.cfg.neural_dataset.lower() != "nsd" and hasattr(self.cfg, 'nsd_type'):
            del self.cfg.nsd_type

        if self.cfg.neural_dataset.lower() == "nsd":
            # Normalize subject_idx to list
            subj = self.cfg.subject_idx
            if isinstance(subj, int):
                subj = [subj]
                self.cfg.subject_idx = subj
            elif isinstance(subj, (list, ListConfig)):
                subj = list(subj)
                self.cfg.subject_idx = subj
            # Validate each element
            for s in subj:
                if not isinstance(s, int) or not 0 <= s < 8:
                    raise AssertionError(
                        f"Invalid subject index for NSD: {s}. Must be an integer in range [0, 7]"
                    )

            # Normalize region to list
            region = self.cfg.region
            if isinstance(region, str):
                region = [region]
                self.cfg.region = region
            elif isinstance(region, (list, ListConfig)):
                region = list(region)
                self.cfg.region = region
            valid_nsd_regions = {"early visual stream", "ventral visual stream"}
            for r in region:
                if r not in valid_nsd_regions:
                    raise AssertionError(
                        f"Invalid region for NSD: {r}. Must be one of {valid_nsd_regions}"
                    )

            # Validate nsd_type
            nsd_type = getattr(self.cfg, "nsd_type", "finegrained")
            if nsd_type not in self.VALID_NSD_TYPES:
                self.rprint(
                    f"[red]Invalid nsd_type: {nsd_type}. Must be in {self.VALID_NSD_TYPES}[/red]",
                    style="error",
                )
                raise AssertionError(f"Invalid nsd_type: {nsd_type}")

        if self.cfg.neural_dataset.lower() == "tvsd":
            # Normalize subject_idx to list
            subj = self.cfg.subject_idx
            if isinstance(subj, int):
                subj = [subj]
                self.cfg.subject_idx = subj
            elif isinstance(subj, (list, ListConfig)):
                subj = list(subj)
                self.cfg.subject_idx = subj
            for s in subj:
                if not isinstance(s, int) or s not in (0, 1):
                    raise AssertionError(
                        f"Invalid subject_idx for TVSD: {s}. Must be 0 (monkey F) or 1 (monkey N)"
                    )

            # Normalize region to list
            region = self.cfg.region
            if isinstance(region, str):
                region = [region]
                self.cfg.region = region
            elif isinstance(region, (list, ListConfig)):
                region = list(region)
                self.cfg.region = region
            valid_regions = {"V1", "V4", "IT"}
            for r in region:
                if r not in valid_regions:
                    raise AssertionError(
                        f"Invalid region for TVSD: {r}. Must be one of {valid_regions}"
                    )

        compare_method = self.cfg.get("compare_method", "spearman").lower()
        if compare_method not in self.VALID_COMPARE_METHODS:
            self.rprint(
                f"[red]Invalid compare_method: {compare_method}. Must be in {self.VALID_COMPARE_METHODS}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid compare_method: {compare_method}")

        if self.cfg.analysis.lower() not in self.VALID_ANALYSES:
            self.rprint(
                f"[red]Invalid analysis: {self.cfg.analysis}. Must be in {self.VALID_ANALYSES}[/red]",
                style="error",
            )
            raise AssertionError(f"Invalid analysis: {self.cfg.analysis}")

        # Encoding score constraints
        if self.cfg.analysis.lower() == "encoding_score":
            if self.cfg.neural_dataset.lower() == "things-behavior":
                raise AssertionError(
                    "analysis=encoding_score is not supported for things-behavior "
                    "(behavioral embeddings have no voxels to predict). Use analysis=rsa instead."
                )
            # Encoding metric is always Pearson r — override whatever the user set
            # (compare_method is an RSA concept). This also ensures run_id hashing
            # uses "pearson" consistently.
            self.cfg.compare_method = "pearson"

        # Model layers validation
        if not hasattr(self.cfg.return_nodes, "__iter__"):
            self.rprint(
                f"[red]return_nodes must be a list-like object[/red]", style="error"
            )
            raise AssertionError("return_nodes must be a list-like object")

        if not self.cfg.return_nodes:
            self.rprint("[red]return_nodes list cannot be empty[/red]", style="error")
            raise AssertionError("return_nodes list cannot be empty")

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
            checkpoint_model_name = self.cfg.checkpoint_model
            seed_letter = get_seed_letter(self.cfg.seed)
            checkpoint_path = Path(
                f"{self.cfg.checkpoint_dir}/cfg{self.cfg.cfg_id}{seed_letter}/{checkpoint_model_name}"
            )
            if not checkpoint_path.exists():
                self.rprint(
                    f"[red]Checkpoint not found: {checkpoint_path}[/red]", style="error"
                )
                raise AssertionError(f"Checkpoint not found: {checkpoint_path}")

        if self.cfg.get("verbose", False):
            self.rprint("✅ Evaluation configuration validation successful", style="success")
        return self.cfg

    def _verify_model_config(self) -> None:
        """Verify model-specific configuration."""
        if self.cfg.model_class == "standard_model":
            if hasattr(self.cfg, "custom_model"):
                self.rprint(
                    "[red]Invalid config: custom_model key present in standard_model mode[/red]",
                    style="error",
                )
                raise AssertionError(
                    "custom_model key should not be present in standard_model mode"
                )
        else:  # custom_model
            if hasattr(self.cfg, "standard_model"):
                self.rprint(
                    "[red]Invalid config: standard_model key present in custom_model mode[/red]",
                    style="error",
                )
                raise AssertionError(
                    "standard_model key should not be present in custom_model mode"
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
    optimizer_name = cfg.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=cfg.learning_rate)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.learning_rate)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def setup_scheduler(optimizer, cfg):
    """Setup learning rate scheduler with warmup support."""
    scheduler_name = cfg.lr_scheduler.lower()
    warmup_epochs = cfg.get("warmup_epochs", 0)
    total_epochs = cfg.num_epochs

    # Calculate T_max for the main scheduler (excluding warmup period)
    T_max = total_epochs - warmup_epochs if warmup_epochs > 0 else total_epochs

    # Define main scheduler based on config
    if scheduler_name == "steplr":
        main_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "multisteplr":
        # Default milestones at 30%, 60%, 90% of training
        default_milestones = [int(T_max * 0.3), int(T_max * 0.6), int(T_max * 0.9)]
        main_scheduler = MultiStepLR(optimizer, milestones=default_milestones, gamma=0.1)
    elif scheduler_name == "cosineannealinglr":
        eta_min = cfg.learning_rate * 0.05
        main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Invalid LR scheduler name: {cfg.lr_scheduler}")

    # Add warmup if configured
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.25,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        return main_scheduler

def get_seed_letter(seed):
    """Convert seed (1-9) to letter (a-i)."""
    if not isinstance(seed, int) or seed < 1 or seed > 9:
        raise ValueError(f"Seed must be an integer between 1-9, got {seed}")
    return chr(ord('a') + seed - 1)
