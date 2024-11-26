import torch.nn as nn
import visreps.models.nn_ops as nn_ops


class CustomCNN(nn.Module):
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
        dropout (bool): Whether to include dropout layers in the classifier. Default is True.
        batchnorm (bool): Whether to include batch normalization layers after each convolutional and linear layer.
                          Default is True.
        pooling_type (str): The type of pooling to use in the feature extractor. Default is 'max'.
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
        
        # Combine all conv layers into a single sequential container
        self.conv = nn.Sequential(
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

        # Sequential container for fc layers
        self.fc = nn.Sequential(
            # fc1
            nn.Flatten(),
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.BatchNorm1d(1024) if batchnorm else nn.Identity(),
            nonlin_fn,
            # fc2
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(1024, 1024),
            nonlin_fn,
            # fc3
            nn.Linear(1024, num_classes)
        )

        # Set trainable parameters based on configuration
        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers):
        """Helper method to set which layers are trainable"""
        # Get conv and fc modules
        conv_modules = [m for m in self.conv.modules() if isinstance(m, nn.Conv2d)]
        fc_modules = [m for m in self.fc.modules() if isinstance(m, nn.Linear)]

        # Set trainable parameters for conv layers
        for idx, module in enumerate(conv_modules):
            module.requires_grad_(trainable_layers["conv"][idx])
            # Find and set corresponding BatchNorm if it exists
            if idx < len(conv_modules) - 1:  # Skip checking after last conv
                next_modules = list(self.conv.modules())[list(self.conv.modules()).index(module) + 1:]
                for next_module in next_modules:
                    if isinstance(next_module, nn.BatchNorm2d):
                        next_module.requires_grad_(True)
                        break

        # Set trainable parameters for fc layers
        for idx, module in enumerate(fc_modules):
            module.requires_grad_(trainable_layers["fc"][idx])
            # Find and set corresponding BatchNorm if it exists
            if idx < len(fc_modules) - 1:  # Skip checking after last fc
                next_modules = list(self.fc.modules())[list(self.fc.modules()).index(module) + 1:]
                for next_module in next_modules:
                    if isinstance(next_module, nn.BatchNorm1d):
                        next_module.requires_grad_(True)
                        break

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x