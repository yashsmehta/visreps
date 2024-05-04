"""
evals.py will be called by runner.py. It is the main flow of control and will call model builder,
benchmarking, and logging results.
"""
from deepjuice import *
import archvision.benchmarker as benchmarker
import torch
import archvision.models.backbone as backbone
import archvision.utils as utils
import archvision.transforms
from pprint import pprint


def eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    cfg = utils.check_and_update_config(cfg)
    match cfg.model.name:
        case "alexnet":
            weights = torch.load('model_checkpoints/alexnet/model_epoch_1.pth')
            model = backbone.CustomAlexNet(num_classes=cfg.num_classes)
            model.load_state_dict(weights)
            print("Loaded weights for AlexNet")
        case "custom":
            model = backbone.VisionModelHack(cfg, device)
        case "alexnet_pretrained":
            model = backbone.AlexNet(pretrained=cfg.model.pretrained)
        case "vgg16":
            model = backbone.VGG16(pretrained=cfg.model.pretrained)
        case "resnet50":
            model = backbone.ResNet50(pretrained=cfg.model.pretrained)
        case "densenet121":
            model = backbone.DenseNet121(pretrained=cfg.model.pretrained)

    preprocess = archvision.transforms.get_preprocess()
    pprint(cfg)

    benchmark = benchmarker.load_benchmark(cfg)
    dataloader = get_data_loader(benchmark.image_paths, preprocess)

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
    max_index = results["model_layer_index"].max()
    results = results[results["model_layer_index"] == max_index]
    if cfg.log_expdata:
        utils.save_logs(results, cfg)

    results = results[
        (results["metric"] == cfg.metric)
        & (results["region"] == cfg.region)
        & (results["cv_split"] == "train")
    ]
    print(results)

