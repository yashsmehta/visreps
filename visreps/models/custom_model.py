import torch
import torch.nn as nn
import math

# Normalization types recognized by isinstance checks
_NORM_TYPES = (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)

# Default number of groups for GroupNorm (Wu & He 2018)
from visreps.models.nn_ops import GN_NUM_GROUPS


class BaseCNN(nn.Module):
    """Base class for custom CNN architectures."""

    def __init__(
        self,
        num_classes=1000,
        trainable_layers=None,
        dropout=0.5,
        pooling_type="max",
        norm_type="group",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.norm_type = norm_type

        self._build_architecture()

        if trainable_layers is not None:
            self._set_trainable_layers(trainable_layers)

        self._initialize_weights()

    def _build_architecture(self):
        raise NotImplementedError

    def _pool(self, kernel_size=3, stride=2):
        if self.pooling_type == "max":
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def _norm(self, channels, spatial=True):
        """Normalization layer. Use spatial=True for conv, False for FC."""
        if self.norm_type == "group":
            return nn.GroupNorm(GN_NUM_GROUPS, channels)
        if self.norm_type == "batch":
            return nn.BatchNorm2d(channels) if spatial else nn.BatchNorm1d(channels)
        raise ValueError(f"Unsupported norm_type: {self.norm_type!r}")

    def _set_trainable_layers(self, trainable_layers):
        """Set trainable layers. Accepts dict with 'conv' and 'fc' as '11100' strings.

        Also freezes associated normalization layers — both their parameters (gamma/beta)
        and, for BatchNorm, their running statistics (kept in eval mode via train() override).
        """
        conv_layers = [m for m in self.features.modules() if isinstance(m, nn.Conv2d)]
        fc_layers = [m for m in self.classifier.modules() if isinstance(m, nn.Linear)]
        conv_norms = [m for m in self.features.modules() if isinstance(m, _NORM_TYPES)]
        fc_norms = [m for m in self.classifier.modules() if isinstance(m, _NORM_TYPES)]

        conv_mask = [c == "1" for c in trainable_layers.get("conv", "1" * len(conv_layers))]
        fc_mask = [c == "1" for c in trainable_layers.get("fc", "1" * len(fc_layers))]

        # Freeze conv/fc weights
        for layer, trainable in zip(conv_layers + fc_layers, conv_mask + fc_mask):
            for p in layer.parameters():
                p.requires_grad = trainable

        # Freeze associated normalization layers (i-th norm corresponds to i-th conv/fc)
        self._frozen_norms = []
        for norm, trainable in zip(conv_norms + fc_norms, conv_mask + fc_mask):
            if not trainable:
                for p in norm.parameters():
                    p.requires_grad = False
                self._frozen_norms.append(norm)

    def train(self, mode=True):
        """Override to keep frozen normalization layers in eval mode.

        GroupNorm has no running statistics, so eval mode is a no-op for it,
        but this override is still needed for BatchNorm compatibility.
        """
        super().train(mode)
        # Check both names: old pickled checkpoints have '_frozen_bns'
        frozen = getattr(self, '_frozen_norms', None)
        if frozen is None:
            frozen = getattr(self, '_frozen_bns', [])
        for norm in frozen:
            norm.eval()
        return self

    def _initialize_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, _NORM_TYPES):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:
                    nn.init.normal_(m.weight, 0, 1.0 / math.sqrt(m.weight.size(1)))
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TinyCustomCNN(BaseCNN):
    """CNN for Tiny ImageNet (64x64 inputs)."""

    def __init__(self, num_classes=200, trainable_layers=None, dropout=0.3,
                 pooling_type="max", norm_type="group"):
        super().__init__(num_classes, trainable_layers, dropout, pooling_type, norm_type)

    def _build_architecture(self):
        self.features = nn.Sequential(
            # conv1: 64 -> 32 -> 16 (after pool)
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            self._norm(64),
            nn.ReLU(inplace=True),
            self._pool(kernel_size=2, stride=2),
            # conv2: 16 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            self._norm(128),
            nn.ReLU(inplace=True),
            # conv3: 16 -> 16 -> 8 (after pool)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            self._norm(256),
            nn.ReLU(inplace=True),
            self._pool(kernel_size=2, stride=2),
            # conv4: 8 -> 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            self._norm(512),
            nn.ReLU(inplace=True),
            # conv5: 8 -> 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            self._norm(512),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(512 * 4 * 4, 2048),
            self._norm(2048, spatial=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(2048, 2048),
            self._norm(2048, spatial=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.num_classes),
        )


class CustomCNN(BaseCNN):
    """AlexNet-style CNN for ImageNet (224x224 inputs)."""

    def __init__(self, num_classes=1000, trainable_layers=None, dropout=0.5,
                 pooling_type="max", norm_type="group"):
        super().__init__(num_classes, trainable_layers, dropout, pooling_type, norm_type)

    def _build_architecture(self):
        self.features = nn.Sequential(
            # conv1: 224 -> 55 -> 27 (after pool)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            self._norm(96),
            nn.ReLU(inplace=True),
            self._pool(kernel_size=3, stride=2),
            # conv2: 27 -> 27 -> 13 (after pool)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=False),
            self._norm(256),
            nn.ReLU(inplace=True),
            self._pool(kernel_size=3, stride=2),
            # conv3: 13 -> 13
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            self._norm(384),
            nn.ReLU(inplace=True),
            # conv4: 13 -> 13
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            self._norm(384),
            nn.ReLU(inplace=True),
            # conv5: 13 -> 13 -> 6 (after pool)
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            self._norm(256),
            nn.ReLU(inplace=True),
            self._pool(kernel_size=3, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 3 * 3, 4096),
            self._norm(4096, spatial=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            self._norm(4096, spatial=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
