import torch.nn as nn
import torch
import archvision.models.nn_ops as nn_ops
from archvision.models.custom_operations.wavelet_conv import WaveletConvolution



class BaseWavelet(nn.Module):

    def __init__(self, num_classes=200, trainable_layers=None, nonlinearity="relu"):
        super(BaseWavelet, self).__init__()
        
        wavelet_layer = WaveletConvolution(filter_type="curvature",device='cuda')
        
        trainable_layers = trainable_layers or {"conv": "01111", "fc": "111"}
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        self.wavelets = nn.Sequential(
            wavelet_layer,
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.features = nn.Sequential(
            nn.Conv2d(wavelet_layer.layer_size, 1024, kernel_size=7, stride=2, padding=2),
            nonlin_fn,
            nn.Conv2d(1024, 2048, kernel_size=5, padding=2),
            nn.BatchNorm2d(2048),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2048, 4096, kernel_size=3, padding=1),
            nn.BatchNorm2d(4096),
            nonlin_fn,
            nn.Conv2d(4096, 1024, kernel_size=3, padding=1),
            nonlin_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1024, 512, kernel_size=2, padding=1),
            nn.BatchNorm2d(512),
            nonlin_fn,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

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
        x = self.wavelets(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
