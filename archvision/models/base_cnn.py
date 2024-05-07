import torch.nn as nn
import torch
import archvision.models.nn_ops as nn_ops
from archvision.models.custom_operations.last import LastLayer

class BaseCNN(nn.Module):
    """
    A basic convolutional neural network model for image classification.

    Attributes:
        features (torch.nn.Sequential): The sequential container of convolutional, batch normalization,
                                        and non-linear activation layers forming the feature extractor part of the CNN.
        avgpool (torch.nn.AdaptiveAvgPool2d): Adaptive average pooling layer to reduce the spatial dimensions
                                              to a fixed size.
        classifier (torch.nn.Sequential): The sequential container of linear layers and dropout layers
                                          forming the classifier part of the CNN.

    Args:
        num_classes (int): Number of output classes for the classifier. Default is 200.
        trainable_layers (dict): A dictionary specifying which layers are trainable. Each key should be
                                 either 'conv' or 'fc', and the corresponding value should be a string of
                                 '1's and '0's indicating the trainability of each layer in the respective
                                 section of the model. Default is all layers trainable.
        nonlinearity (str): The type of nonlinearity to use. Default is 'relu'.
    """

    def __init__(self, num_classes=200, trainable_layers=None, nonlinearity="relu", dropout=True, batchnorm=True):
        super(BaseCNN, self).__init__()
        trainable_layers = trainable_layers or {"conv": "11111", "fc": "111"}
        trainable_layers = {layer_type: [val == "1" for val in layers] for layer_type, layers in trainable_layers.items()}

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nonlin_fn,
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            *(batchnorm and [nn.BatchNorm2d(64)] or []),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            *(batchnorm and [nn.BatchNorm2d(128)] or []),
            nonlin_fn,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=2, padding=1),
            *(batchnorm and [nn.BatchNorm2d(512)] or []),
            nonlin_fn
        ]
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        classifier_layers = [
            nn.Linear(512 * 3 * 3, 1024),
            *(dropout and [nn.Dropout(inplace=True)] or []),
            *(batchnorm and [nn.BatchNorm1d(1024)] or []),
            nonlin_fn,
            nn.Linear(1024, 1024),
            *(dropout and [nn.Dropout(inplace=True)] or []),
            nonlin_fn,
            nn.Linear(1024, num_classes)
        ]
        self.classifier = nn.Sequential(*classifier_layers)
        self.last = LastLayer()
        self.last.__class__.__name__ = "last_layer"

        conv_idx = 0
        fc_idx = 0
        print("Trainable layers: ", trainable_layers)

        for module in self.features:
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
            elif isinstance(module, nn.Conv2d):
                module.requires_grad_(trainable_layers["conv"][conv_idx])
                conv_idx += 1

        for module in self.classifier:
            if isinstance(module, nn.BatchNorm1d):
                module.requires_grad_(True)
            elif isinstance(module, nn.Linear):
                module.requires_grad_(trainable_layers["fc"][fc_idx])
                fc_idx += 1

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor containing the image data.

        Returns:
            torch.Tensor: The output tensor containing the class probabilities.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.last(x)
        return x
