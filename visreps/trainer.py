import time
import torch
import torch.nn as nn
import torch.optim as optim
from visreps.dataloader import tinyimgnet_loader
from visreps.models.custom_cnn import CustomCNN
from visreps.models.standard_cnns import AlexNet, VGG16, ResNet50, DenseNet121
import visreps.utils as utils
import wandb
from omegaconf import OmegaConf
import json
import os
from tqdm import tqdm

def train(cfg):
    """
    train a cnn model based on configuration.
    args:
        cfg (omegaconf.dictconfig): training configuration parameters
    returns:
        model: trained model
    """
    cfg = utils.check_trainer_config(cfg)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    data_loader = tinyimgnet_loader(
        batchsize=cfg.batchsize,
        data_augment=cfg.data_augment
    )

    model_classes = {
        "custom_cnn": CustomCNN,
        "alexnet": AlexNet,
        "vgg16": VGG16, 
        "resnet50": ResNet50,
        "densenet121": DenseNet121
    }

    if cfg.model_class not in model_classes:
        raise ValueError(f"Invalid model: {cfg.model_class}. Choose from {list(model_classes.keys())}")

    model_params = {
        "num_classes": cfg.num_classes,
        **({"trainable_layers": {"conv": cfg.custom.conv_trainable, "fc": cfg.custom.fc_trainable},
            "nonlinearity": cfg.custom.nonlinearity,
            "dropout": cfg.custom.dropout, 
            "batchnorm": cfg.custom.batchnorm,
            "pooling_type": cfg.custom.pooling_type} if cfg.model_class == "custom_cnn"
           else {"pretrained": cfg.pretrained})
    }

    model = model_classes[cfg.model_class](**model_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for training")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.exp_name,
            group=cfg.group,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"seed_{cfg.seed}",
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if cfg.log_checkpoints:
        checkpoint_subdir = utils.make_checkpoint_dir(folder=cfg.exp_name)

    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        total_steps = len(data_loader["train"])
        
        progress_bar = tqdm(data_loader["train"], desc=f"Epoch {epoch}/{cfg.num_epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / total_steps
        epoch_duration = time.time() - start_time
        
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

        model.eval()
        with torch.no_grad():
            test_acc = utils.calculate_accuracy(data_loader["test"], model, device)
            print(f"Test Accuracy after epoch {epoch}: {test_acc:.2f}%")
            if cfg.use_wandb:
                wandb.log({"epoch": epoch, "test_acc": test_acc})

            if cfg.evaluate_train:
                train_acc = utils.calculate_accuracy(data_loader["train"], model, device)
                print(f"Train Accuracy after epoch {epoch}: {train_acc:.2f}%")
                if cfg.use_wandb:
                    wandb.log({"epoch": epoch, "train_acc": train_acc})

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
