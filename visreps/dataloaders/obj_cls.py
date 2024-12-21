import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os

DS_MEAN = {"tiny-imagenet": [0.480, 0.448, 0.398], "imgnet": [0.485, 0.456, 0.406]}
DS_STD = {"tiny-imagenet": [0.272, 0.265, 0.274], "imgnet": [0.229, 0.224, 0.225]}

def get_transform(
    ds_stats="imgnet",
    data_augment=False,
    image_size=224,
):
    """
    Get a transform for an image dataset. This is the standard AlexNet transform for ImageNet1K.
    """
    transform_list = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size)
    ]
    if data_augment:
        transform_list.extend(
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        )
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=DS_MEAN[ds_stats], std=DS_STD[ds_stats])]
    )
    
    transform = transforms.Compose(transform_list)
    return transform


def tinyimgnet_loader(batchsize=32, num_workers=8, data_augment=True, ds_stats="tiny-imagenet"):
    assert ds_stats in ["tiny-imagenet", "imgnet"]
    data_transform = {
        "train": get_transform(ds_stats=ds_stats, data_augment=data_augment),
        "test": get_transform(ds_stats=ds_stats, data_augment=False)
    }

    base_dir = os.path.join("data", "tiny-imagenet-200")
    
    datasets_dict = {}
    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, "train" if split == "train" else "val")
        try:
            dataset = datasets.ImageFolder(split_dir, data_transform[split])
            datasets_dict[split] = dataset
        except Exception as e:
            raise

    dataloaders = {}
    for split, dataset in datasets_dict.items():
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, 
                              num_workers=num_workers, prefetch_factor=2)
        dataloaders[split] = dataloader

    return dataloaders
