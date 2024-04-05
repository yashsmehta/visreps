"""
evals.py will be called by runner.py. It is the main flow of control and will call model builder,
benchmarking, and logging results.
"""
from deepjuice import *
import archvision.benchmarker as benchmarker
import archvision.models as models


def run(cfg):
    benchmark = benchmarker.load_benchmark(cfg)

    model = models.builder(cfg)
    preprocess = models.preprocess(cfg)
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
