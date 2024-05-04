import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from archvision.transforms import get_transform
from archvision.models import nn_ops
import wandb


def calculate_accuracy(data_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=200, trainable_layers=None, nonlinearity="relu"):
        super(CustomAlexNet, self).__init__()
        trainable_layers = trainable_layers or {"conv": "11111", "fc": "111"}
        trainable_layers = {
            layer_type: [val == '1' for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nonlin_fn,
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nonlin_fn,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=2, padding=1),
            nn.BatchNorm2d(512),
            nonlin_fn,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        conv_idx = 0
        fc_idx = 0
        print("Trainable layers: ", trainable_layers)

        for module in self.features:
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
            elif isinstance(module, nn.Conv2d):
                module.requires_grad_(trainable_layers["conv"][conv_idx])
                conv_idx += 1

        for module in self.classifier:
            if isinstance(module, nn.BatchNorm1d):
                module.requires_grad_(True)
            elif isinstance(module, nn.Linear):
                module.requires_grad_(trainable_layers["fc"][fc_idx])
                fc_idx += 1

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


DS_MEAN = [0.480, 0.448, 0.398]
DS_STD = [0.272, 0.265, 0.274]
NUM_CLASSES = 200

if __name__ == "__main__":
    batch_size = 64
    num_epochs = 50
    use_wandb = False
    checkpoint_interval = 10
    seed = 1
    torch.manual_seed(seed)

    trainable_layers = {"conv": "10000", "fc": "001"}

    data_transform = {
        "train": get_transform(data_augmentation=True, mean=DS_MEAN, std=DS_STD),
        "test": get_transform(data_augmentation=False, mean=DS_MEAN, std=DS_STD),
    }

    train_dataset = datasets.ImageFolder(
        root="data/tiny-imagenet-200/train", transform=data_transform["train"]
    )
    test_dataset = datasets.ImageFolder(
        root="data/tiny-imagenet-200/val", transform=data_transform["test"]
    )

    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            prefetch_factor=2,
        ),
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
        ),
    }
    model = CustomAlexNet(num_classes=NUM_CLASSES, trainable_layers=trainable_layers)

    # model = models.alexnet()
    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    print(model)
    print(
        "Total trainable params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Using {'GPU' if device == 'cuda' else 'CPU'} for training")

    if use_wandb:
        wandb.init(project="alexnet_partial")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
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
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader["train"])}], Loss: {loss.item():.6f}'
                )

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {time.time() - start:.2f}s")

        model.eval()
        with torch.no_grad():
            if (epoch + 1) % checkpoint_interval == 0:
                train_acc = calculate_accuracy(data_loader["train"], model, device)
                print(f"Train Accuracy: {train_acc:.2f}%")
                checkpoint_dir = f"model_checkpoints/alexnet/{seed}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved as {model_path}")

            test_acc = calculate_accuracy(data_loader["test"], model, device)
            print(f"Test Accuracy: {test_acc:.2f}%")

            if use_wandb:
                train_acc = train_acc if (epoch + 1) % checkpoint_interval == 0 else "N/A"
                wandb.log(
                    {
                        "epoch": epoch,
                        "loss": loss.item(),
                        "test_acc": test_acc,
                        "train_acc": train_acc,
                    }
                )

