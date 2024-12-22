import time
import torch
import torch.nn as nn
import torch.optim as optim
from visreps.dataloaders.obj_cls import tinyimgnet_loader
from visreps.models.custom_cnn import CustomCNN
from visreps.models.standard_cnns import AlexNet, VGG16, ResNet50, DenseNet121
import visreps.utils as utils
import wandb
from omegaconf import OmegaConf
import json
import os
from tqdm import tqdm
from visreps.metrics import calculate_cls_accuracy

def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        return calculate_cls_accuracy(loader, model, device)

def train(cfg):
    """Train a model for object classification"""
    cfg = utils.check_trainer_config(cfg)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    data_loader = tinyimgnet_loader(cfg.batchsize, cfg.data_augment)

    # Initialize model
    model_classes = {
        "custom_cnn": CustomCNN,
        "alexnet": AlexNet,
        "vgg16": VGG16,
        "resnet50": ResNet50,
        "densenet121": DenseNet121,
    }
    if cfg.model_class not in model_classes:
        raise ValueError(f"Invalid model: {cfg.model_class}")

    if cfg.model_class == "custom_cnn":
        params = {
            "num_classes": cfg.num_classes,
            "trainable_layers": {"conv": cfg.custom.conv_trainable, "fc": cfg.custom.fc_trainable},
            "nonlinearity": cfg.custom.nonlinearity,
            "dropout": cfg.custom.dropout,
            "batchnorm": cfg.custom.batchnorm,
            "pooling_type": cfg.custom.pooling_type,
        }
    else:
        params = {"num_classes": cfg.num_classes, "pretrained": cfg.pretrained}

    model = model_classes[cfg.model_class](**params)
    model.to(device)

    # Initialize wandb if needed
    if cfg.use_wandb:
        wandb.init(project=cfg.exp_name, 
                  group=cfg.group, 
                  config=OmegaConf.to_container(cfg, resolve=True), 
                  name=f"seed_{cfg.seed}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Save initial checkpoint if needed
    if cfg.log_checkpoints:
        checkpoint_subdir = utils.make_checkpoint_dir(cfg.exp_name)
        cfg_dict = {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **OmegaConf.to_container(cfg, resolve=True),
        }
        with open(os.path.join(checkpoint_subdir, "config.json"), "w") as f:
            json.dump(cfg_dict, f)
        torch.save(model.state_dict(), os.path.join(checkpoint_subdir, "model_init.pth"))
        print(f"Model saved to {checkpoint_subdir}")

    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Train one epoch
        for images, labels in tqdm(data_loader["train"], desc=f"Epoch {epoch}/{cfg.num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Evaluate
        test_acc = evaluate(model, data_loader["test"], device)
        metrics = {
            "epoch": epoch, 
            "loss": running_loss / len(data_loader["train"]), 
            "test_acc": test_acc
        }

        if cfg.evaluate_train:
            metrics["train_acc"] = evaluate(model, data_loader["train"], device)

        # Logging
        if cfg.use_wandb:
            wandb.log(metrics)

        if cfg.log_checkpoints and epoch % cfg.checkpoint_interval == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(checkpoint_subdir, f"model_epoch_{epoch:02d}.pth")
            )

        if epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs:
            print(f"Epoch [{epoch}/{cfg.num_epochs}] "
                  f"Loss: {metrics['loss']:.6f}, "
                  f"Test Acc: {metrics['test_acc']:.2f}%, "
                  f"Time: {time.time() - start_time:.2f}s")

    print("Training complete.")
    if cfg.use_wandb:
        wandb.finish()
    return model