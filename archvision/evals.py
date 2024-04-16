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
    # model = backbone.VisionModel(cfg, device)
    # model = backbone.AlexNet(pretrained=False)
    model = backbone.VGGNet(pretrained=False)
    preprocess = archvision.transforms.get_preprocess()
    pprint(cfg)

    benchmark = benchmarker.load_benchmark(cfg)
    dataloader = get_data_loader(benchmark.image_paths, preprocess)

    devices = {"device": device, "output_device": "cpu"}

    extractor = FeatureExtractor(
        model, dataloader, flatten=True, batch_progress=True, keep=["last_layer"], **devices
    )

    all_results = benchmarker.get_benchmarking_results(benchmark, extractor)
    filtered_results = all_results[(all_results['metric'] == cfg.metric) & (all_results['region'] == cfg.region) & (all_results['cv_split'] == 'train')]
    max_score_result = filtered_results.loc[filtered_results['score'].idxmax()]
    print(max_score_result)
