import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from archvision.transforms import get_transform
from archvision.alex_imgnet import CustomAlexNet, calculate_accuracy
import archvision.utils as utils
import wandb
from omegaconf import OmegaConf
import json
import os


DS_MEAN = [0.480, 0.448, 0.398]
DS_STD = [0.272, 0.265, 0.274]

def make_checkpoint_dir(cfg):
    checkpoint_dir = os.path.join("model_checkpoints", cfg.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ith_folder = len(os.listdir(checkpoint_dir)) + 1
    checkpoint_subdir = os.path.join(checkpoint_dir, "cfg"+str(ith_folder))
    os.makedirs(checkpoint_subdir, exist_ok=True)
    return checkpoint_subdir

def train(cfg):
    cfg = utils.check_trainer_config(cfg)
    torch.manual_seed(cfg.seed)
    trainable_layers = {"conv": cfg.conv_trainable, "fc": cfg.fc_trainable}

    data_transform = {
        "train": get_transform(data_augment=cfg.data_augment, mean=DS_MEAN, std=DS_STD),
        "test": get_transform(data_augment=False, mean=DS_MEAN, std=DS_STD),
    }

    train_dataset = datasets.ImageFolder(
        root=cfg.data_dir + "/train", transform=data_transform["train"]
    )
    test_dataset = datasets.ImageFolder(
        root=cfg.data_dir + "/val", transform=data_transform["test"]
    )

    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=cfg.batchsize,
            shuffle=True,
            num_workers=8,
            prefetch_factor=2,
        ),
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=cfg.batchsize,
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
        ),
    }
    model = CustomAlexNet(num_classes=cfg.num_classes, trainable_layers=trainable_layers)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", total_params)
    print("Trainable params: ", trainable_params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using {'GPU' if device == 'cuda' else 'CPU'} for training")

    if cfg.use_wandb:
        wandb.init(project=cfg.exp_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if cfg.log_checkpoints:
        checkpoint_subdir = make_checkpoint_dir(cfg)

    for epoch in range(cfg.num_epochs):

        model.eval()
        with torch.no_grad():
            if (epoch) % cfg.checkpoint_interval == 0:
                train_acc = calculate_accuracy(data_loader["train"], model, device)
                print(f"Train Accuracy: {train_acc:.2f}%")
                if cfg.log_checkpoints:
                    model_path = os.path.join(checkpoint_subdir, f"model_epoch_{epoch:02d}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved as {model_path}")
            test_acc = calculate_accuracy(data_loader["test"], model, device)
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
                print(f'Epoch [{epoch+1}/{cfg.num_epochs}], Step [{i+1}/{len(data_loader["train"])}], Loss: {loss.item():.6f}')

                if cfg.use_wandb:
                    train_acc = train_acc if (epoch + 1) % cfg.checkpoint_interval == 0 else "N/A"
                    wandb.log({
                        "epoch": epoch,
                        "loss": loss.item(),
                        "test_acc": test_acc,
                        "train_acc": train_acc,
                    })

        epoch_duration = time.time() - start
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Time: {epoch_duration:.2f}s")

    if cfg.log_checkpoints:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict['epoch_duration'] = epoch_duration
        cfg_dict['total_params'] = total_params
        cfg_dict['trainable_params'] = trainable_params

        with open(f"{checkpoint_subdir}/config.json", 'w') as f:
            json.dump(cfg_dict, f)