from deepjuice import *
import archvision.benchmarker as benchmarker
import archvision.dataloader
import archvision.utils
from archvision.models.standard_cnns import AlexNet
import torch
import json


def eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model_checkpoint_path = f'model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}'
    with open(f'{model_checkpoint_path}/config.json', 'r') as f:
        training_config = json.load(f)

    model = torch.load(f'{model_checkpoint_path}/model_epoch_{cfg.epoch:02d}.pth')
    print(f"Loaded model for config id: {cfg.cfg_id} for epoch: {cfg.epoch}")
    # model = AlexNet(pretrained=True)
    print(model)

    benchmark = benchmarker.load_benchmark(cfg)
    dataloader = get_data_loader(benchmark.image_paths, archvision.dataloader.get_transform(image_size=224))

    devices = {"device": device, "output_device": "cpu"}
    print(f"'Keep' layer for deepjuice: {cfg.deepjuice_keep_layer}")
    keep = [] if cfg.deepjuice_keep_layer == "all" else [cfg.deepjuice_keep_layer]

    extractor = FeatureExtractor(model, dataloader, flatten=True, batch_progress=True, keep=keep, **devices)
    results = benchmarker.get_benchmarking_results(benchmark, extractor)
    results = results[results["region"] == cfg.region]
    results['epoch'] = cfg.epoch
    print(results[results["metric"] == "srpr"])

    for key, value in training_config.items():
        results[key] = value

    if cfg.log_expdata:
        archvision.utils.log_results(results, file_name=cfg.exp_name)

