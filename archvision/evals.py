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
    model = backbone.VisionModel(cfg, device)
    preprocess = archvision.transforms.get_preprocess()
    pprint(cfg)
    print(model)

    benchmark = benchmarker.load_benchmark(cfg)
    dataloader = get_data_loader(benchmark.image_paths, preprocess)

    devices = {"device": device, "output_device": "cpu"}

    extractor = FeatureExtractor(
        model, dataloader, flatten=True, batch_progress=True, **devices
    )

    results = benchmarker.get_benchmarking_results(benchmark, extractor)

    results = results.sort_values(by="model_layer_index")
    results = results[results["region"] == cfg.voxel_set]
    results = results[results["metric"] == cfg.eval_metric]
    print(results)
