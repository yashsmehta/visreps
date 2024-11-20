import time
import torch
import torch.nn as nn
import torch.optim as optim
from visreps.dataloader import tinyimgnet_loader
from visreps.models.base_cnn import BaseCNN
import visreps.utils as utils
import wandb
from omegaconf import OmegaConf
import json
import os
from tqdm import tqdm

def train(cfg):
    """
    Train a convolutional neural network model based on the provided configuration.

    Args:
        cfg (OmegaConf.DictConfig): A configuration object containing all necessary parameters.
    """
    # Validate and update the training configuration.
    cfg = utils.check_trainer_config(cfg)
    torch.manual_seed(cfg.seed)

    # Define trainable layers.
    trainable_layers = {"conv": cfg.conv_trainable, "fc": cfg.fc_trainable}

    # Get data loaders.
    data_loader = tinyimgnet_loader(cfg.batchsize)

    # Select the model class.
    model_classes = {
        "base_cnn": BaseCNN,
        # Add other model classes here if available.
    }

    if cfg.model_class not in model_classes:
        raise ValueError(f"Model class '{cfg.model_class}' not recognized.")

    model = model_classes[cfg.model_class](
        num_classes=cfg.num_classes,
        trainable_layers=trainable_layers,
        nonlinearity=cfg.nonlinearity,
        dropout=cfg.dropout,
        batchnorm=cfg.batchnorm,
        pooling_type=cfg.pooling_type,
    )
    print(model)

    # Calculate total and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training")

    # Initialize Weights & Biases if enabled.
    if cfg.use_wandb:
        wandb.init(
            project=cfg.exp_name,
            group=cfg.group,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"seed_{cfg.seed}",
        )

    # Set up loss function and optimizer.
    criterion = nn.CrossEntropyLoss()

    # Make optimizer type and learning rate configurable.
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{cfg.optimizer}' not recognized.")

    # Set up checkpoint directory if logging checkpoints.
    if cfg.log_checkpoints:
        checkpoint_subdir = utils.make_checkpoint_dir(folder=cfg.exp_name)

    # Training loop.
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        # Training phase.
        model.train()
        running_loss = 0.0
        total_steps = len(data_loader["train"])
        
        # Progress bar for batches
        progress_bar = tqdm(data_loader["train"], desc=f"Epoch {epoch}/{cfg.num_epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass.
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss.
            running_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate average loss for the epoch
        avg_loss = running_loss / total_steps
        epoch_duration = time.time() - start_time
        
        # Log every log_interval epochs
        if epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs:
            print(
                f"\nEpoch [{epoch}/{cfg.num_epochs}], "
                f"Average Loss: {avg_loss:.6f}, "
                f"Time: {epoch_duration:.2f}s"
            )
            if cfg.use_wandb:
                wandb.log(
                    {"epoch": epoch, "loss": avg_loss},
                )
        running_loss = 0.0

        # Evaluation phase.
        model.eval()
        with torch.no_grad():
            test_acc = utils.calculate_accuracy(data_loader["test"], model, device)
            print(f"Test Accuracy after epoch {epoch}: {test_acc:.2f}%")
            if cfg.use_wandb:
                wandb.log({"epoch": epoch, "test_acc": test_acc})

            # Optionally evaluate on training data.
            if cfg.evaluate_train:
                train_acc = utils.calculate_accuracy(data_loader["train"], model, device)
                print(f"Train Accuracy after epoch {epoch}: {train_acc:.2f}%")
                if cfg.use_wandb:
                    wandb.log({"epoch": epoch, "train_acc": train_acc})

        # Save checkpoints.
        if cfg.log_checkpoints and epoch % cfg.checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': OmegaConf.to_container(cfg, resolve=True),
            }
            model_path = os.path.join(
                checkpoint_subdir, f"model_epoch_{epoch:02d}.pth"
            )
            torch.save(checkpoint, model_path)
            print(f"Model checkpoint saved at {model_path}")

    # Save final configuration and stats.
    if cfg.log_checkpoints:
        cfg_dict = {
            "last_epoch_duration": epoch_duration,
            "total_params": total_params,
            "trainable_params": trainable_params,
            **OmegaConf.to_container(cfg, resolve=True),
        }
        with open(f"{checkpoint_subdir}/config.json", "w") as f:
            json.dump(cfg_dict, f)

    # Finish wandb run.
    if cfg.use_wandb:
        wandb.finish()
