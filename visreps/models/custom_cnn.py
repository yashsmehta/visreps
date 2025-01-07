import torch
import torch.nn as nn
import visreps.models.nn_ops as nn_ops


class CustomCNN(nn.Module):
    """
    A basic convolutional neural network model for image classification, structured similar to AlexNet.

    Attributes:
        features (nn.Sequential): The convolutional and pooling layers
        avgpool (nn.AdaptiveAvgPool2d): Adaptive pooling layer
        classifier (nn.Sequential): The fully connected layers
    """

    def __init__(
        self,
        num_classes=200,
        trainable_layers=None,
        nonlinearity="relu",
        dropout=True,
        batchnorm=True,
        pooling_type="max",
    ):
        super(CustomCNN, self).__init__()
        trainable_layers = trainable_layers or {"conv": "11111", "fc": "111"}
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        # Get activation and pooling functions
        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(pooling_type)

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=not batchnorm),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,
            # conv2
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=not batchnorm),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nonlin_fn,
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,
            # conv4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nonlin_fn,
            # conv5
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nonlin_fn,
        )

        self.adaptive_pool = nn_ops.get_pooling_fn("adaptive" + pooling_type)

        self.classifier = nn.Sequential(
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.BatchNorm1d(1024) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(1024, 1024),
            nonlin_fn,
            nn.Linear(1024, num_classes),
        )

        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers):
        """Helper method to set which layers are trainable"""
        # Get conv and fc modules
        conv_modules = [m for m in self.features.modules() if isinstance(m, nn.Conv2d)]
        fc_modules = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]

        # Set trainable parameters for conv layers
        for idx, module in enumerate(conv_modules):
            module.requires_grad_(trainable_layers["conv"][idx])
            # Find and set corresponding BatchNorm if it exists
            if idx < len(conv_modules) - 1:  # Skip checking after last conv
                next_modules = list(self.features.modules())[
                    list(self.features.modules()).index(module) + 1 :
                ]
                for next_module in next_modules:
                    if isinstance(next_module, nn.BatchNorm2d):
                        next_module.requires_grad_(True)
                        break

        # Set trainable parameters for fc layers
        for idx, module in enumerate(fc_modules):
            module.requires_grad_(trainable_layers["fc"][idx])
            # Find and set corresponding BatchNorm if it exists
            if idx < len(fc_modules) - 1:  # Skip checking after last fc
                next_modules = list(self.classifier.modules())[
                    list(self.classifier.modules()).index(module) + 1 :
                ]
                for next_module in next_modules:
                    if isinstance(next_module, nn.BatchNorm1d):
                        next_module.requires_grad_(True)
                        break

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
