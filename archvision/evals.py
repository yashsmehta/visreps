"""
evals.py will be called by runner.py. It is the main flow of control and will call model builder,
benchmarking, and logging results.
"""
from deepjuice import *
import archvision.benchmarker as benchmarker
import archvision.models.backbone as backbone
import archvision.transforms as transforms


def run(cfg):

    benchmark = benchmarker.load_benchmark(cfg)

    # model = backbone.VisionModel(cfg)
    # preprocess = preprocessor.get_preprocess()

    model_uid = 'torchvision_alexnet_imagenet1k_v1'
    model, preprocess = get_deepjuice_model(model_uid)

    dataloader = get_data_loader(benchmark.image_paths, preprocess)

    devices = {"device": "cuda:0", "output_device": "cpu"}

    extractor = FeatureExtractor(
        model, dataloader, flatten=True, batch_progress=True, **devices
    )

    results = benchmarker.get_benchmarking_results(benchmark, extractor)

    results = results.sort_values(by="model_layer_index")
    results = results[results['region'] == cfg.voxel_set]
    results = results[results['metric'] == cfg.eval_metric]
    print(results)
