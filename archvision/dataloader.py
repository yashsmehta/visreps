import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os


DS_MEAN = {"tiny-imagenet": [0.480, 0.448, 0.398], "imgnet": [0.485, 0.456, 0.406]}
DS_STD = {"tiny-imagenet": [0.272, 0.265, 0.274], "imgnet": [0.229, 0.224, 0.225]}


def get_transform(
    mean=DS_MEAN["imgnet"], std=DS_STD["imgnet"], data_augment=False, image_size=64
):
    """
    This function returns a composition of image transformations.

    Args:
        mean (list): The mean values for normalization.
        std (list): The standard deviation values for normalization.
        data_augment (bool): If True, apply data augmentation.
        image_size (int): The size to which the image will be resized.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.
    """
    transform_list = [transforms.Resize(image_size)]
    if data_augment:
        transform_list.extend(
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        )
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return transforms.Compose(transform_list)


def get_dataloader(data_dir, batchsize, num_workers=8, ds_stats="tiny-imagenet"):
    """
    This function returns a dictionary of data loaders for the specified dataset.

    Args:
        data_dir (str): The directory where the dataset is located.
        batchsize (int): The batch size for the data loaders.
        num_workers (int): The number of worker processes to use for data loading.
        ds_stats (str): The dataset statistics to use for normalization.

    Returns:
        dict: A dictionary of data loaders for the specified dataset.
    """
    assert ds_stats in ["tiny-imagenet", "imgnet"]
    data_transform = {
        "train": get_transform(
            mean=DS_MEAN[ds_stats], std=DS_STD[ds_stats], data_augment=True
        ),
        "test": get_transform(
            mean=DS_MEAN[ds_stats], std=DS_STD[ds_stats], data_augment=False
        ),
    }

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=data_transform["train"]
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=data_transform["test"]
    )

    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2,
        ),
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
        ),
    }
    return data_loader
