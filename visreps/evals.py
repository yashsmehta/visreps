import visreps.benchmarker as benchmarker
from visreps.dataloader import get_data_loader
import visreps.utils as utils
import torch
import json
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from visreps.models.standard_cnns import AlexNet, VGG16, ResNet50, DenseNet121


def eval(cfg):
    """
    Evaluate a model using a specified configuration.

    This function performs the following steps:
    1. Determines the device to use based on CUDA availability.
    2. Constructs the path to the model checkpoint.
    3. Loads the training configuration from the checkpoint.
    4. Loads the model from the checkpoint.
    5. Loads the benchmark using the configuration.
    6. Gets the data loader with the appropriate transformations.
    7. Sets up device configuration for feature extraction.
    8. Determines layers to keep based on configuration.
    9. Initializes the feature extractor with the model and dataloader.
    10. Gets benchmarking results and filters by region.
    11. Appends training configuration to results.
    12. Logs results if specified in configuration.

    Args:
        cfg (OmegaConf): Configuration object containing experiment details and settings.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if cfg.load_checkpoint:
        model_checkpoint_path = f"model_checkpoints/{cfg.exp_name}/cfg{cfg.cfg_id}"
        with open(f"{model_checkpoint_path}/config.json", "r") as f:
            training_config = json.load(f)

        model = torch.load(f"{model_checkpoint_path}/model_epoch_{cfg.epoch:02d}.pth")
    else:
        if cfg.model == "alexnet":
            model = create_feature_extractor(AlexNet(pretrained=cfg.pretrained), return_nodes={'classifier.6': 'fc8'})
        elif cfg.model == "vgg16":
            model = create_feature_extractor(VGG16(pretrained=cfg.pretrained), return_nodes={'classifier.6': 'fc8'})
        elif cfg.model == "resnet50":
            model = create_feature_extractor(ResNet50(pretrained=cfg.pretrained), return_nodes={'fc': 'fc8'})
        elif cfg.model == "densenet121":
            model = create_feature_extractor(DenseNet121(pretrained=cfg.pretrained), return_nodes={'classifier': 'fc8'})

        print(model)
        exit()

        # Interface with NSD data
        nsd_data = utils.load_pickle('data/nsd/neural_responses.pkl')
        selected_images = utils.load_pickle('data/nsd/stimuli.pkl')
        dataloader = get_data_loader(selected_images, utils.get_transform(image_size=64))

        # Extract activations
        activations_dict = {}
        model.eval()
        with torch.no_grad():
            for image_ids, images in dataloader:
                outputs = model(images)['fc8']
                for image_id, activation in zip(image_ids, outputs):
                    activations_dict[int(image_id)] = activation.cpu().numpy()

        benchmark = benchmarker.load_benchmark(cfg)

        devices = {"device": device, "output_device": "cpu"}

        try:
            results = benchmarker.get_benchmarking_results(benchmark, activations_dict)
            results = results[results["region"] == cfg.region]
            results["epoch"] = cfg.epoch
            print(results[results["metric"] == "srpr"])

            for key, value in training_config.items():
                results[key] = value

            if cfg.log_expdata:
                utils.log_results(results, folder_name=cfg.exp_name, cfg_id=cfg.cfg_id)

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except KeyError as e:
            print(f"Key error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
