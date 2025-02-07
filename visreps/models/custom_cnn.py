import torch
import torch.nn as nn
import visreps.models.nn_ops as nn_ops
import math


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
        # Default all layers to trainable if not specified
        trainable_layers = trainable_layers or {"conv": "11111", "fc": "111"}
        
        # Validate trainable_layers length matches architecture
        n_conv_layers = 5  # We have 5 conv layers
        n_fc_layers = 3    # We have 3 fc layers
        
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
        pool_fn = nn_ops.get_pooling_fn(pooling_type)

        # Adjusted architecture for Tiny ImageNet (64x64 input)
        self.features = nn.Sequential(
            # conv1: 64x64 -> 32x32
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,  # 32x32
            
            # conv2: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,  # 16x16
            
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
            nn.Dropout(0.3) if dropout else nn.Identity(),  # Reduced dropout
            nn.Linear(512 * 4 * 4, 2048),
            nn.BatchNorm1d(2048) if batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout(0.3) if dropout else nn.Identity(),  # Reduced dropout
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512) if batchnorm else nn.Identity(),  # Added BN
            nonlin_fn,
            nn.Linear(512, num_classes),
        )

        self._set_trainable_layers(trainable_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization with fan_out mode for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use larger initialization for final layers
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
        # Feature extraction
        x = self.features(x)
        
        # Debug shape before pooling
        if not hasattr(self, '_shape_checked'):
            print(f"Feature shape before pooling: {x.shape}")
            self._shape_checked = True
            
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Debug shape after pooling
        if not hasattr(self, '_pool_checked'):
            print(f"Shape after adaptive pooling: {x.shape}")
            self._pool_checked = True
            
        # Flatten for classifier
        x = torch.flatten(x, 1)
        
        # Debug shape before classifier
        if not hasattr(self, '_flatten_checked'):
            print(f"Shape after flattening: {x.shape}")
            self._flatten_checked = True
            
        # Classification
        x = self.classifier(x)
        return x
