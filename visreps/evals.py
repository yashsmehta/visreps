import torch
import numpy as np
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from visreps.dataloader import get_dataloader, get_transform
import visreps.utils as utils
import visreps.benchmarker as benchmarker
import visreps.metrics as metrics

def load_model(cfg):
    if cfg.load_checkpoint:
        model = utils.load_checkpoint(cfg.checkpoint_path)
    else:
        model_map = {
            "alexnet": models.alexnet,
            "vgg16": models.vgg16,
            "resnet50": models.resnet50,
            "densenet121": models.densenet121,
        }
        if cfg.model.lower() not in model_map:
            raise ValueError(
                f"Model {cfg.model} not supported. Choose from: {list(model_map.keys())}"
            )
        model = model_map[cfg.model.lower()](pretrained=cfg.pretrained)

    # Define default return nodes if not specified in cfg
    default_return_nodes = {
        "alexnet": ["features.12"],
        "vgg16": ["features.29"],
        "resnet50": ["layer4"],
        "densenet121": ["features.denseblock4"],
    }
    return_nodes = getattr(cfg, "return_nodes", default_return_nodes[cfg.model.lower()])

    # Create a feature extractor
    return_nodes_dict = {node_name: node_name for node_name in return_nodes}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes_dict)
    return feature_extractor, {}

def extract_activations(model, dataloader, device):
    model.eval()
    activations_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs)
            for node_name, output in outputs.items():
                output_np = output.cpu().numpy()
                activations_dict.setdefault(node_name, []).append(output_np)
    # Concatenate activations for each layer
    for node_name in activations_dict:
        activations_dict[node_name] = np.concatenate(activations_dict[node_name], axis=0)
    return activations_dict

def eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, training_config = load_model(cfg)
    model.to(device)

    neural_data, stimuli = benchmarker.load_benchmark_data(cfg)
    dataloader = get_dataloader(stimuli, get_transform(image_size=64))
    activations_dict = extract_activations(model, dataloader, device)
    results = metrics.calculate_rsa_score(neural_data, activations_dict)

    results["epoch"] = cfg.epoch
    print(f"RSA Score: {results['rsa_scores']}")
    if cfg.load_checkpoint:
        results.update(training_config)
    if cfg.log_expdata:
        try:
            utils.log_results(results, folder_name=cfg.exp_name, cfg_id=cfg.cfg_id)
        except (FileNotFoundError, KeyError) as e:
            print(f"An error occurred while logging results: {e}")
