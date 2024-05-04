from deepjuice import *
import archvision.benchmarker as benchmarker
import torch
import time
import random
from pathlib import Path
import archvision.models.backbone as backbone
from archvision.transforms import get_transform
from archvision.alex_imgnet import CustomAlexNet
import json


def eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    model_checkpoint_path = f'model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}'
    with open(f'{model_checkpoint_path}/config.json', 'r') as f:
        training_config = json.load(f)

    weights = torch.load(f'{model_checkpoint_path}/model_epoch_{cfg.epoch:02d}.pth')
    model = CustomAlexNet(num_classes=cfg.num_classes)
    # model = backbone.AlexNet(pretrained=True)
    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, cfg.num_classes)
    print("Loaded weights for CustomAlexNet model for epoch: ", cfg.epoch)
    model.load_state_dict(weights)
    print(model)

    benchmark = benchmarker.load_benchmark(cfg)
    dataloader = get_data_loader(benchmark.image_paths, get_transform())

    devices = {"device": device, "output_device": "cpu"}
    print("'Keep' layer for deepjuice: ", cfg.model.deepjuice_keep_layer)
    keep = (
        []
        if cfg.model.deepjuice_keep_layer == "all"
        else [cfg.model.deepjuice_keep_layer]
    )

    extractor = FeatureExtractor(
        model, dataloader, flatten=True, batch_progress=True, keep=keep, **devices
    )

    results = benchmarker.get_benchmarking_results(benchmark, extractor)
    results = results[
        (results["region"] == cfg.region)
    ]
    results['epoch'] = cfg.epoch
    for key, value in training_config.items():
        results[key] = value

    # max_index = results["model_layer_index"].max()
    # results = results[results["model_layer_index"] == max_index]
    if cfg.log_expdata:
        time.sleep(random.uniform(1, 10))
        logdata_path = Path("logs/")
        logdata_path.mkdir(parents=True, exist_ok=True)
        csv_file = logdata_path / f"partial_training.csv"
        write_header = not csv_file.exists()

        # Use a lock file to synchronize access to the CSV file
        lock_file = csv_file.with_suffix(".lock")
        while lock_file.exists():
            print(f"Waiting for lock on {csv_file}...")
            time.sleep(random.uniform(1, 5))

        try:
            lock_file.touch()
            results.to_csv(csv_file, mode="a", header=write_header, index=False)
            print(f"Saved logs to {csv_file}")
        finally:
            lock_file.unlink()


