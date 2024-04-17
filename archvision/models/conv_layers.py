import torch.nn as nn
import models.nn_ops as nn_ops


class ConvolutionLayers(nn.Module):
    def __init__(self, cfg, in_channels, device):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.n_channels = (
            in_channels
            if cfg.model.conv_in_channels == "none"
            else cfg.model.conv_in_channels
        )
        self.device = device

        for i, layer in enumerate(cfg.model.layers):
            conv_layer = nn.Conv2d(
                in_channels, layer.channels, kernel_size=layer.kernel_size, padding=1
            ).to(self.device)
            nn_ops.initialize_weights(conv_layer, cfg.model.weights_init, cfg.model.seed)

            nonlinearity = nn_ops.get_nonlinearity(cfg.model.nonlin)
            normalization = nn_ops.get_normalization(layer.channels, cfg.model.norm)
            pooling = nn_ops.get_pooling(layer.pooling, layer.pool_kernel_size)
            if i == len(cfg.model.layers) - 1:
                conv_layer.__class__.__name__ = "last_layer"

            self.conv_layers.append(
                nn.Sequential(conv_layer, nonlinearity, normalization, pooling)
            )
            in_channels = layer.channels

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.conv_layers:
            x = layer(x)
        return x
