import time
import torch
import torch.nn as nn
import torch.optim as optim
from archvision.dataloader import get_dataloader
from archvision.models.base_cnn import BaseCNN
from archvision.models.wavelet_front import WaveletNet
from archvision.models.scat_transform_front import ScatTransformNet
from archvision.models.scat_transform_base_cnn import ScatTransformBaseCNN
from archvision.models.scat_transform_custom import ScatTransformCustom

import archvision.utils as utils
import wandb
from omegaconf import OmegaConf
import json
import os


def train(cfg):
    """
    Train a convolutional neural network model based on the provided configuration.

    This function sets up and trains a CNN model using the specified training configuration.
    It initializes the model, sets up data loaders, defines the loss function and optimizer,
    and manages the training process including logging and checkpointing.

    Args:
        cfg (OmegaConf.DictConfig): A configuration object containing all necessary parameters
                                    for training, including data paths, model parameters, training
                                    hyperparameters, and logging settings.

    The function performs the following steps:
    1. Validates and updates the training configuration.
    2. Sets the random seed for reproducibility.
    3. Initializes the model with specified trainable layers.
    4. Moves the model to the appropriate device (CPU or GPU).
    5. Optionally initializes Weights & Biases logging.
    6. Sets up the loss function and optimizer.
    7. Enters a training loop which includes:
        - Evaluating the model on the test set and optionally on the training set.
        - Saving model checkpoints at specified intervals.
        - Training the model on the training data.
        - Logging loss and accuracy metrics.
    8. Optionally saves the final model configuration and statistics.

    Outputs:
        The function does not return any values but will output logs directly to the console
        and optionally to Weights & Biases. It also saves model checkpoints and configuration
        to the disk based on the provided settings.
    """
    cfg = utils.check_trainer_config(cfg)
    torch.manual_seed(cfg.seed)
    trainable_layers = {"conv": cfg.conv_trainable, "fc": cfg.fc_trainable}
    data_loader = get_dataloader(cfg.data_dir, cfg.batchsize, ds_stats="tiny-imagenet")

    model_class = {
        "base_cnn": BaseCNN,
        "wavelet_net": WaveletNet,
        "scattransform_net": ScatTransformNet,
        "scattransform_base_cnn":ScatTransformBaseCNN,
        "scattransform_custom":ScatTransformCustom
        
    }[cfg.model_class]

    model = model_class(
        num_classes=cfg.num_classes,
        trainable_layers=trainable_layers,
        nonlinearity=cfg.nonlinearity,
        dropout=cfg.dropout,
        batchnorm=cfg.batchnorm,
        pooling_type=cfg.pooling_type,
    )
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", total_params)
    print("Trainable params: ", trainable_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using {'GPU' if device == 'cuda' else 'CPU'} for training")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.exp_name,
            group=cfg.group,
            config=dict(cfg),
            name=f"seed_{cfg.seed}",
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if cfg.log_checkpoints:
        checkpoint_subdir = utils.make_checkpoint_dir(folder=cfg.exp_name)

    for epoch in range(cfg.num_epochs + 1):
        model.eval()
        with torch.no_grad():
            if epoch % cfg.checkpoint_interval == 0:
                train_acc = utils.calculate_accuracy(
                    data_loader["train"], model, device
                )
                print(f"Train Accuracy: {train_acc:.2f}%")
                if cfg.log_checkpoints:
                    model_path = os.path.join(
                        checkpoint_subdir, f"model_epoch_{epoch:02d}.pth"
                    )
                    torch.save(model, model_path)
                    print(f"Model saved as {model_path}")
            test_acc = utils.calculate_accuracy(data_loader["test"], model, device)
            print(f"Test Accuracy: {test_acc:.2f}%")

        start = time.time()
        model.train()
        for i, (images, labels) in enumerate(data_loader["train"]):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{cfg.num_epochs}], Step [{i+1}/{len(data_loader["train"])}], Loss: {loss.item():.6f}'
                )
                if cfg.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "test_acc": test_acc,
                            "train_acc": (
                                train_acc
                                if epoch % cfg.checkpoint_interval == 0
                                else "N/A"
                            ),
                        }
                    )

        epoch_duration = time.time() - start
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Time: {epoch_duration:.2f}s")

    if cfg.log_checkpoints:
        cfg_dict = {
            "epoch_duration": epoch_duration,
            "total_params": total_params,
            "trainable_params": trainable_params,
            **OmegaConf.to_container(cfg, resolve=True),
        }
        with open(f"{checkpoint_subdir}/config.json", "w") as f:
            json.dump(cfg_dict, f)
