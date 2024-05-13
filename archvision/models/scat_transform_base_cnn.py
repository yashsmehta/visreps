import torch
import torch.nn as nn
import archvision.models.nn_ops as nn_ops
from archvision.models.custom_operations.scat_transform import ScatTransform


class ScatTransformBaseCNN(nn.Module):


    def __init__(
        self,
        num_classes=200,
        trainable_layers=None,
        nonlinearity="relu",
        dropout=True,
        batchnorm=True,
        pooling_type="max",
    ):
        super(ScatTransformBaseCNN, self).__init__()
        trainable_layers = trainable_layers or {"conv": "0000", "fc": "001"}
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(pooling_type)

        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nonlin_fn,
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64) if batchnorm else None,
            nonlin_fn,
            pool_fn,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else None,
            nonlin_fn,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nonlin_fn,
            pool_fn
        ]
        layers = [layer for layer in layers if layer is not None]
        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn_ops.get_pooling_fn("adaptive" + pooling_type)
        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)

        self.scattransform = ScatTransform(C = 256, J=3, L=4, M=8, N=8)

        classifier_layers = [
            nn.Dropout() if dropout else None,
            nn.Linear(self.scattransform.layer_size, 1024),
            nn.BatchNorm1d(1024) if batchnorm else None,
            nonlin_fn,
            nn.Dropout() if dropout else None,
            nn.Linear(1024, 1024),
            nonlin_fn,
            nn.Linear(1024, num_classes),
        ]
        classifier_layers = [layer for layer in classifier_layers if layer is not None]
        self.classifier = nn.Sequential(*classifier_layers)

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
        x = self.features(x)
        #x = self.adaptive_pool(x)
        x = self.scattransform(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x