import torch
import torch.nn as nn
import visreps.models.nn_ops as nn_ops
import math


class BaseCNN(nn.Module):
    """
    Base class for custom CNN architectures with shared functionality.

    Subclasses should implement _build_architecture() to define features and classifier.
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
        super(BaseCNN, self).__init__()
        self.num_classes = num_classes
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.pooling_type = pooling_type

        # Build architecture (implemented by subclass)
        self._build_architecture()

        # Set trainable layers
        if trainable_layers is not None:
            trainable_layers = self._normalize_trainable_layers(trainable_layers)
            self._set_trainable_layers(trainable_layers)

        # Initialize weights
        self._initialize_weights()

    def _build_architecture(self):
        """
        Build the network architecture. Must be implemented by subclasses.
        Should set: self.features, self.adaptive_pool, self.classifier
        """
        raise NotImplementedError("Subclasses must implement _build_architecture()")

    def _normalize_trainable_layers(self, trainable_layers):
        """
        Normalize trainable_layers to dict of boolean lists with correct lengths.
        Accepts strings like "11100" or boolean lists like [True, True, True, False, False].
        """
        # Count actual layers in the architecture
        n_conv = len([m for m in self.features.modules() if isinstance(m, nn.Conv2d)])
        n_fc = len([m for m in self.classifier.modules() if isinstance(m, nn.Linear)])

        normalized = {}
        for layer_type, n_layers in [("conv", n_conv), ("fc", n_fc)]:
            layers = trainable_layers.get(layer_type, "1" * n_layers)

            # Convert string to boolean list if needed
            if isinstance(layers, str):
                layers = [c == "1" for c in layers]

            # Validate length
            if len(layers) != n_layers:
                print(f"Warning: {layer_type} trainable_layers has length {len(layers)}, "
                      f"expected {n_layers}. Adjusting...")
                # Pad with True or truncate
                layers = (layers + [True] * n_layers)[:n_layers]

            normalized[layer_type] = layers

        return normalized

    def _set_trainable_layers(self, trainable_layers):
        """Set which layers are trainable based on boolean lists."""
        conv_modules = [m for m in self.features.modules() if isinstance(m, nn.Conv2d)]
        fc_modules = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]

        # Set trainable for conv layers and their BatchNorms
        for idx, conv in enumerate(conv_modules):
            trainable = trainable_layers["conv"][idx]
            for param in conv.parameters():
                param.requires_grad = trainable

        # Set trainable for fc layers
        for idx, fc in enumerate(fc_modules):
            trainable = trainable_layers["fc"][idx]
            for param in fc.parameters():
                param.requires_grad = trainable

        # Always keep BatchNorm trainable when its layer is trainable
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LocalResponseNorm)):
                for param in module.parameters():
                    param.requires_grad = True

    def _initialize_weights(self):
        """Initialize model weights with He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                # Final layer: smaller std, hidden layers: He init
                if m.out_features == self.num_classes:
                    std = 1.0 / math.sqrt(m.weight.size(1))
                    nn.init.normal_(m.weight, 0, std)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TinyCustomCNN(BaseCNN):
    """
    A basic convolutional neural network for Tiny ImageNet (64x64 inputs).
    Uses an AlexNet-style architecture adapted for smaller images.
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
        super(TinyCustomCNN, self).__init__(
            num_classes=num_classes,
            trainable_layers=trainable_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            batchnorm=batchnorm,
            pooling_type=pooling_type,
        )

    def _build_architecture(self):
        """Build architecture for Tiny ImageNet (64x64 inputs)."""
        nonlin_fn = nn_ops.get_nonlinearity(self.nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(self.pooling_type, kernel_size=2, stride=2)

        self.features = nn.Sequential(
            # conv1: 64x64 -> 16x16
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=not self.batchnorm),
            nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,

            # conv2: 16x16 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not self.batchnorm),
            nn.BatchNorm2d(128) if self.batchnorm else nn.Identity(),
            nonlin_fn,

            # conv3: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=not self.batchnorm),
            nn.BatchNorm2d(256) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,

            # conv4: 8x8 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=not self.batchnorm),
            nn.BatchNorm2d(512) if self.batchnorm else nn.Identity(),
            nonlin_fn,

            # conv5: 8x8 -> 8x8
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=not self.batchnorm),
            nn.BatchNorm2d(512) if self.batchnorm else nn.Identity(),
            nonlin_fn,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.BatchNorm1d(2048) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Linear(2048, self.num_classes),
        )


class CustomCNN(BaseCNN):
    """
    AlexNet-style CNN adapted for ImageNet (224x224 inputs).
    Uses grouped convolutions and local response normalization.
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
        super(CustomCNN, self).__init__(
            num_classes=num_classes,
            trainable_layers=trainable_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
            batchnorm=batchnorm,
            pooling_type=pooling_type,
        )

    def _build_architecture(self):
        """Build architecture for ImageNet (224x224 inputs)."""
        nonlin_fn = nn_ops.get_nonlinearity(self.nonlinearity, inplace=True)
        pool_fn = nn_ops.get_pooling_fn(self.pooling_type, kernel_size=3, stride=2)

        self.features = nn.Sequential(
            # conv1: 224x224 -> 56x56 (after pooling)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=not self.batchnorm),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0) if not self.batchnorm else nn.BatchNorm2d(96),
            nonlin_fn,
            pool_fn,

            # conv2: 56x56 -> 28x28 (after pooling)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2, bias=not self.batchnorm),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0) if not self.batchnorm else nn.BatchNorm2d(256),
            nonlin_fn,
            pool_fn,

            # conv3: 28x28 -> 14x14
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=not self.batchnorm),
            nn.BatchNorm2d(384) if self.batchnorm else nn.Identity(),
            nonlin_fn,

            # conv4: 14x14 -> 14x14
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=not self.batchnorm),
            nn.BatchNorm2d(384) if self.batchnorm else nn.Identity(),
            nonlin_fn,

            # conv5: 14x14 -> 7x7 (after pooling)
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=not self.batchnorm),
            nn.BatchNorm2d(256) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            pool_fn,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096) if self.batchnorm else nn.Identity(),
            nonlin_fn,
            nn.Linear(4096, self.num_classes),
        )