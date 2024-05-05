from omegaconf import OmegaConf
import time
import random
from pathlib import Path
import torch
import os


def check_and_update_config(cfg):
    """
    asserts that the configuration is valid, and adds the default layer values.
    """
    assert cfg.model.name in [
        "alexnet",
        "vgg16",
        "resnet50",
        "densenet121",
        "custom",
    ], "only vgg16, resnet50, densenet121, custom supported!"
    assert cfg.model.deepjuice_keep_layer in [
        "last_layer",
        "all",
    ], "only last_layer, all supported for deepjuice keep layer!"
    assert cfg.model.pretrained in [
        True,
        False,
    ], "pretrained must be True or False!"

    if cfg.model.name == "custom":
        cfg.model.pretrained = False
        assert cfg.model.nonlin in [
            "none",
            "relu",
            "elu",
            "tanh",
            "sigmoid",
        ], "only linear, elu, relu, tanh, sigmoid supported!"
        assert cfg.model.weights_init in [
            "kaiming",
            "kaiming_uniform",
            "xavier",
            "xavier_uniform",
            "gaussian",
            "uniform",
        ], "only kaiming, kaiming_uniform, xavier, xavier_uniform, gaussian, uniform supported!"
        assert cfg.model.norm in [
            "batch",
            "instance",
            "channel",
            "none",
        ], "only batch, instance, channel, none supported!"
        default_layer = {
            "kernel_size": 3,
            "channels": 64,
            "pooling": "none",
            "pool_kernel_size": 2,
        }
        for i, layer in enumerate(cfg.model.layers):
            default_layer.update(layer)
            cfg.model.layers[i] = default_layer.copy()

            assert cfg.model.layers[i].pooling in [
                "max",
                "globalmax",
                "avg",
                "globalavg",
                "none",
            ], "only max, avg, globalmax, globalavg and none supported!"
            if layer.pooling in ["globalmax", "globalavg"]:
                cfg.model.layers[i].pool_kernel_size = "N/A"
    else:
        cfg.model = OmegaConf.create(
            {
                "name": cfg.model.name,
                "pretrained": cfg.model.pretrained,
                "deepjuice_keep_layer": cfg.model.deepjuice_keep_layer,
            }
        )

    return cfg

def check_trainer_config(cfg):
    assert len(cfg.conv_trainable) == 5, "conv_trainable must have 5 elements!"
    assert len(cfg.fc_trainable) == 3, "fc_trainable must have 3 elements!"
    assert all(char in '01' for char in cfg.conv_trainable), "conv_trainable must only contain 0s and 1s!"
    assert all(char in '01' for char in cfg.fc_trainable), "fc_trainable must only contain 0s and 1s!"
    return cfg


def save_logs(df, cfg):
    df = df.drop(columns=["model_layer_index"])
    df["exp_name"] = cfg.exp_name
    df["seed"] = cfg.seed
    if cfg.model.name != "custom":
        df["model"] = cfg.model.name
        df["pretrained"] = cfg.model.pretrained
    else:
        df["nonlin"] = cfg.model.nonlin
        df["weights_init"] = cfg.model.weights_init
        df["norm"] = cfg.model.norm

    logdata_path = Path(cfg.log_dir)
    if cfg.model.name == "custom":
        logdata_path = logdata_path / "custom_arch" 
    else:
        logdata_path = logdata_path / "standard_arch" 

    logdata_path.mkdir(parents=True, exist_ok=True)
    csv_file = logdata_path / f"{cfg.exp_name}.csv"
    write_header = not csv_file.exists()

    # Use a lock file to synchronize access to the CSV file
    lock_file = csv_file.with_suffix(".lock")
    while lock_file.exists():
        print(f"Waiting for lock on {csv_file}...")
        time.sleep(random.uniform(1, 5))

    try:
        lock_file.touch()
        df.to_csv(csv_file, mode="a", header=write_header, index=False)
        print(f"Saved logs to {csv_file}")
    finally:
        lock_file.unlink()

    return logdata_path


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


def make_checkpoint_dir(folder, parent_dir='model_checkpoints'):
    checkpoint_dir = os.path.join(parent_dir, folder)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ith_folder = len(os.listdir(checkpoint_dir)) + 1
    checkpoint_subdir = os.path.join(checkpoint_dir, "cfg"+str(ith_folder))
    os.makedirs(checkpoint_subdir, exist_ok=True)
    return checkpoint_subdir

def log_results(results, file_name):
    logdata_path = Path("logs/")
    logdata_path.mkdir(parents=True, exist_ok=True)
    csv_file = logdata_path / f"{file_name}.csv"
    write_header = not csv_file.exists()

    lock_file = csv_file.with_suffix(".lock")
    try:
        while lock_file.exists():
            print(f"Waiting for lock on {csv_file}...")
            time.sleep(random.uniform(1, 5))
        lock_file.touch(exist_ok=False)

        results.to_csv(csv_file, mode="a", header=write_header, index=False)
        print(f"Saved logs to {csv_file}")
    except FileExistsError:
        print(f"Lock file already exists, skipping logging to {csv_file}")
    finally:
        if lock_file.exists():
            lock_file.unlink()

