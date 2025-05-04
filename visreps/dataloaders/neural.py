import os, logging
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import xarray as xr

import visreps.utils as utils
from visreps.dataloaders.obj_cls import get_transform
from bonner.datasets.hebart2022_things_behavior._data import (
    load_embeddings as _load_things_embed,
)
from bonner.datasets.hebart2019_things._stimuli import StimulusSet as _ThingsStimSet

logger = logging.getLogger(__name__)


# ───────────────────────── NSD ──────────────────────────
def load_nsd_data(cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load fMRI responses and corresponding stimulus images for a specified NSD subject and brain region.

    Args:
        cfg (Dict): Configuration dictionary with keys:
            - "region" (str): Brain region name.
            - "subject_idx" (int): Subject index.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            - Dictionary mapping stimulus IDs (str) to fMRI response arrays (np.ndarray).
            - Dictionary mapping stimulus IDs (str) to image arrays (np.ndarray).
            Only stimulus IDs present in both fMRI and image data are included.
    """
    region, subj = cfg["region"], cfg["subject_idx"]
    root = utils.get_env_var("NSD_DATA_DIR")
    fmri_xr = utils.load_pickle(os.path.join(root, "fmri_responses.pkl"))[region][subj]
    images = {
        str(k): v
        for k, v in utils.load_pickle(os.path.join(root, "stimuli.pkl")).items()
    }
    ids = {str(i) for i in fmri_xr.coords["stimulus"].values} & images.keys()
    return (
        {i: fmri_xr.sel(stimulus=int(i)).values for i in ids},
        {i: images[i] for i in ids},
    )


# ─────────────────── NSD Synthetic ────────────────────
def load_nsd_synthetic_data(
    cfg: Dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load synthetic‑NSD fMRI responses and matching stimuli.

    Args:
        cfg (Dict): {"region": str, "subject_idx": int}

    Returns:
        (targets, stimuli):
            - targets: {stim_id: np.ndarray}
            - stimuli: {stim_id: PIL.Image.Image}
    """
    region, subj = cfg["region"], cfg["subject_idx"]
    root = utils.get_env_var("NSD_SYNTHETIC_DATA_DIR")

    fmri = utils.load_pickle(os.path.join(root, "fmri_responses.pkl"))[region][subj]
    images = {
        str(k): v
        for k, v in utils.load_pickle(os.path.join(root, "stimuli.pkl")).items()
    }

    ids = {str(k) for k in fmri} & images.keys()

    return (
        {i: fmri[i] for i in ids},
        {i: images[i] for i in ids},
    )


# ──────────────────────── THINGS ────────────────────────
def load_things_data() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Load THINGS behavioral embeddings and corresponding image file paths.

    Returns:
        embeds (dict[str, np.ndarray]): Mapping from object ID to embedding vector (float32).
        img_paths (dict[str, str]): Mapping from object ID to absolute image file path.
            Only includes images with valid, existing paths as resolved via _ThingsStimSet logic.
    """
    beh_xr = _load_things_embed()
    embeds = {
        str(i): beh_xr.sel(object=i).values.astype(np.float32)
        for i in beh_xr["object"].values
    }

    img_paths = {}
    stimset = _ThingsStimSet()
    for k in stimset.metadata.index:
        try:
            relative_filename = stimset.metadata.loc[k, "filename"]
            relative_path_part = relative_filename.split("images/", 1)[-1]
            correct_path = os.path.join(
                stimset.root, "images", "object_images", relative_path_part
            )
            if os.path.exists(correct_path):
                img_paths[k] = correct_path
            else:
                logger.warning(
                    f"Constructed path {correct_path} for image {k} does not exist. Skipping."
                )
        except KeyError:
            logger.error(f"Key {k} not found in _ThingsStimSet metadata. Skipping.")
        except Exception as e:
            logger.warning(
                f"Error constructing path for image {k}: {e}. Skipping.", exc_info=False
            )

    return embeds, img_paths


# ─────────────────────── Dataset/Loader ───────────────────────
class _StimuliDataset(Dataset):
    """
    PyTorch Dataset for stimuli, supporting both file paths and in-memory image data.

    Args:
        stimuli (Dict[str, Any]): Mapping from stimulus ID to file path, np.ndarray, or PIL.Image.
        transform (callable): Transform to apply to each image.
    """

    def __init__(self, stimuli: Dict[str, Any], transform):
        self.keys = sorted(stimuli.keys())
        self.stimuli = {k: stimuli[k] for k in self.keys}
        self.tr = transform or transforms.ToTensor()

    def __len__(self):
        """Return number of stimuli."""
        return len(self.keys)

    def _load_and_transform(self, data_or_path: Any, key: str):
        """
        Load and transform an image from a path, np.ndarray, or PIL.Image.
        Returns a transformed tensor or a placeholder on error.
        """
        img = None
        path_used = None
        try:
            if isinstance(data_or_path, str):
                path_used = data_or_path
                with Image.open(path_used) as opened_img:
                    img = opened_img.convert("RGB")
            elif isinstance(data_or_path, np.ndarray):
                img = Image.fromarray(data_or_path.astype("uint8"), "RGB")
            elif isinstance(data_or_path, Image.Image):
                img = data_or_path
            else:
                logger.warning(
                    f"Unexpected data type {type(data_or_path)} for key {key}. Returning placeholder."
                )

            if img is not None:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return self.tr(img)
        except FileNotFoundError:
            logger.warning(
                f"Image file not found for key {key} at path {path_used}. Returning placeholder."
            )
        except OSError as e:
            err_loc = f"at path {path_used}" if path_used else "from data"
            if "truncated" in str(e):
                logger.warning(
                    f"OSError (truncated) processing image key {key} {err_loc}. Applying transform to placeholder."
                )
            else:
                logger.error(
                    f"Unhandled OSError for key {key} {err_loc}: {e}. Returning placeholder."
                )
        except Exception as e:
            err_loc = f"at path {path_used}" if path_used else "from data"
            logger.error(
                f"Unexpected error for key {key} {err_loc}: {e}. Returning placeholder."
            )

        try:
            target_size = (224, 224)
            if hasattr(self.tr, "transforms"):
                for t in self.tr.transforms:
                    if isinstance(
                        t,
                        (
                            transforms.Resize,
                            transforms.CenterCrop,
                            transforms.RandomResizedCrop,
                        ),
                    ):
                        size = getattr(t, "size", target_size)
                        if isinstance(size, int):
                            size = (size, size)
                        if isinstance(size, tuple) and len(size) == 2:
                            target_size = size
                            break
            placeholder = Image.new("RGB", target_size, color="grey")
            return self.tr(placeholder)
        except Exception as placeholder_e:
            logger.error(
                f"Failed to create/transform placeholder for key {key}: {placeholder_e}"
            )
            return torch.zeros((3, 224, 224))

    def __getitem__(self, idx):
        key = self.keys[idx]
        data_or_path = self.stimuli[key]
        transformed_img = self._load_and_transform(data_or_path, key)
        return transformed_img, key


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[str]]:
    imgs, keys = zip(*batch)
    return torch.stack(imgs), list(keys)


def _make_loader(stimuli, transform, batch, workers):
    return DataLoader(
        _StimuliDataset(stimuli, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def get_neural_loader(cfg: Dict) -> Tuple[Dict[str, Any], DataLoader]:
    """
    Returns (targets, dataloader) for the specified neural dataset.
    Supported datasets: 'nsd', 'things', 'nsd_synthetic'.
    """
    dataset_type = cfg.get("neural_dataset")
    if dataset_type == "nsd":
        targets, stimuli = load_nsd_data(cfg)
    elif dataset_type == "things":
        targets, stimuli = load_things_data()
    elif dataset_type == "nsd_synthetic":
        targets, stimuli = load_nsd_synthetic_data(cfg)
    else:
        raise ValueError("neural_dataset must be 'nsd', 'things', or 'nsd_synthetic'")

    transform = get_transform(ds_stats="imgnet")
    dataloader = _make_loader(
        stimuli,
        transform,
        cfg["batchsize"],
        cfg["num_workers"],
    )
    return targets, dataloader
