import torchvision.transforms as transforms


def get_transform(data_augment=False, size=64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform_list = [transforms.Resize(size)]
    if data_augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms.Compose(transform_list)


def get_transform_eval():
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess