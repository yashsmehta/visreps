import torch
import torch.nn as nn
import visreps.models.nn_ops as nn_ops
import math


class TinyCustomCNN(nn.Module):
    """
    A basic convolutional neural network model for Tiny ImageNet classification.
    Uses a AlexNet-style architecture adapted for 64x64 inputs.

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
        dropout=0.3,
        batchnorm=True,
        pooling_type="max",
    ):
        super(TinyCustomCNN, self).__init__()
        
        # Validate trainable_layers length matches architecture
        n_conv_layers = 5  # We have 5 conv layers
        n_fc_layers = 3    # We have 3 fc layers
        self.num_classes = num_classes

        # Ensure conv_trainable has correct length
        if len(trainable_layers["conv"]) < n_conv_layers:
            print(f"Warning: conv_trainable length {len(trainable_layers['conv'])} is less than number of conv layers {n_conv_layers}")
            trainable_layers["conv"] = trainable_layers["conv"].ljust(n_conv_layers, "1")
        elif len(trainable_layers["conv"]) > n_conv_layers:
            print(f"Warning: conv_trainable length {len(trainable_layers['conv'])} is more than number of conv layers {n_conv_layers}")
            trainable_layers["conv"] = trainable_layers["conv"][:n_conv_layers]
            
        # Ensure fc_trainable has correct length
        if len(trainable_layers["fc"]) < n_fc_layers:
            print(f"Warning: fc_trainable length {len(trainable_layers['fc'])} is less than number of fc layers {n_fc_layers}")
            trainable_layers["fc"] = trainable_layers["fc"].ljust(n_fc_layers, "1")
        elif len(trainable_layers["fc"]) > n_fc_layers:
            print(f"Warning: fc_trainable length {len(trainable_layers['fc'])} is more than number of fc layers {n_fc_layers}")
            trainable_layers["fc"] = trainable_layers["fc"][:n_fc_layers]
        
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        # Get activation and pooling functions
        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(pooling_type, kernel_size=2, stride=2)

        # Adjusted architecture for Tiny ImageNet (64x64 input)
        self.features = nn.Sequential(
            # conv1: 64x64 -> 16x16
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=not batchnorm),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,  # 16x16

            # conv2: 16x16 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nonlin_fn,

            # conv3: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,  # 8x8

            # conv4: 8x8 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nonlin_fn,

            # conv5: 8x8 -> 8x8
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nonlin_fn,
        )

        # Use 4x4 adaptive pooling for better feature extraction
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Adjusted classifier with better scaling
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout else nn.Identity(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.BatchNorm1d(2048) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout(p=dropout) if dropout else nn.Identity(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Linear(2048, num_classes),
        )

        self._set_trainable_layers(trainable_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization for Conv layers (ReLU networks)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Check if this is the final layer by comparing out_features to num_classes.
                if m.out_features == self.num_classes:
                    # Final classification layer: use a smaller std.
                    fan_in = m.weight.size(1)
                    std = 1.0 / math.sqrt(fan_in)
                    nn.init.normal_(m.weight, 0, std)
                else:
                    # Hidden layers: use He initialization (gain for ReLU).
                    gain = nn.init.calculate_gain('relu')
                    fan_in = m.weight.size(1)
                    std = gain / math.sqrt(fan_in)
                    nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        """Forward pass through the network."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomCNN(nn.Module):
    """
    An AlexNet-style CNN architecture adapted for image classification.
    Uses grouped convolutions and local response normalization as in the original AlexNet.

    Attributes:
        features (nn.Sequential): The convolutional and pooling layers
        avgpool (nn.AdaptiveAvgPool2d): Adaptive pooling layer
        classifier (nn.Sequential): The fully connected layers
    """

    def __init__(
        self,
        num_classes=1000,
        trainable_layers=None,
        nonlinearity="relu",
        dropout=0.5,
        batchnorm=True,
        pooling_type="max",
    ):
        super(CustomCNN, self).__init__()
        
        # Validate trainable_layers length matches architecture
        n_conv_layers = 5  # We have 5 conv layers
        n_fc_layers = 3    # We have 3 fc layers
        self.num_classes = num_classes

        # Ensure conv_trainable has correct length
        if len(trainable_layers["conv"]) < n_conv_layers:
            print(f"Warning: conv_trainable length {len(trainable_layers['conv'])} is less than number of conv layers {n_conv_layers}")
            trainable_layers["conv"] = trainable_layers["conv"].ljust(n_conv_layers, "1")
        elif len(trainable_layers["conv"]) > n_conv_layers:
            print(f"Warning: conv_trainable length {len(trainable_layers['conv'])} is more than number of conv layers {n_conv_layers}")
            trainable_layers["conv"] = trainable_layers["conv"][:n_conv_layers]
            
        # Ensure fc_trainable has correct length
        if len(trainable_layers["fc"]) < n_fc_layers:
            print(f"Warning: fc_trainable length {len(trainable_layers['fc'])} is less than number of fc layers {n_fc_layers}")
            trainable_layers["fc"] = trainable_layers["fc"].ljust(n_fc_layers, "1")
        elif len(trainable_layers["fc"]) > n_fc_layers:
            print(f"Warning: fc_trainable length {len(trainable_layers['fc'])} is more than number of fc layers {n_fc_layers}")
            trainable_layers["fc"] = trainable_layers["fc"][:n_fc_layers]
        
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        # Get activation and pooling functions
        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(pooling_type, kernel_size=3, stride=2)

        # Adjusted architecture for Tiny ImageNet (64x64 input) following AlexNet structure
        self.features = nn.Sequential(
            # conv1: 64x64 -> 15x15 (after pooling)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=not batchnorm),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0) if not batchnorm else nn.BatchNorm2d(96),
            nonlin_fn,
            pool_fn,  # 15x15

            # conv2: 15x15 -> 7x7 (after pooling)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2, bias=not batchnorm),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0) if not batchnorm else nn.BatchNorm2d(256),
            nonlin_fn,
            pool_fn,  # 7x7

            # conv3: 7x7 -> 7x7
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(384) if batchnorm else nn.Identity(),
            nonlin_fn,

            # conv4: 7x7 -> 7x7
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=not batchnorm),
            nn.BatchNorm2d(384) if batchnorm else nn.Identity(),
            nonlin_fn,

            # conv5: 7x7 -> 3x3 (after pooling)
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=not batchnorm),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,  # 3x3
        )

        # Use 3x3 output size to match AlexNet's final feature map dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # Classifier matching AlexNet's structure with 4096 neurons
        self.classifier = nn.Sequential(
            nn.Dropout(0.5) if dropout else nn.Identity(),  # Original AlexNet dropout
            nn.Linear(256 * 3 * 3, 4096),
            nn.BatchNorm1d(4096) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout(0.5) if dropout else nn.Identity(),  # Original AlexNet dropout
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Linear(4096, num_classes),
        )

        self._set_trainable_layers(trainable_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization for Conv layers (ReLU networks)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Check if this is the final layer by comparing out_features to num_classes.
                if m.out_features == self.num_classes:
                    # Final classification layer: use a smaller std.
                    fan_in = m.weight.size(1)
                    std = 1.0 / math.sqrt(fan_in)
                    nn.init.normal_(m.weight, 0, std)
                else:
                    # Hidden layers: use He initialization (gain for ReLU).
                    gain = nn.init.calculate_gain('relu')
                    fan_in = m.weight.size(1)
                    std = gain / math.sqrt(fan_in)
                    nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
                    if isinstance(next_module, (nn.BatchNorm2d, nn.LocalResponseNorm)):
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
        """Forward pass through the network."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
