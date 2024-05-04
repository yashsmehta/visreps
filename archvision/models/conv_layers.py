import torch.nn as nn
import models.nn_ops as nn_ops


class ConvolutionLayers(nn.Module):
    def __init__(self, cfg, in_channels, device):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.n_channels = in_channels if cfg.model.conv_in_channels == "none" else cfg.model.conv_in_channels
        self.device = device

        for layer in cfg.model.layers:
            self.add_layer(cfg, in_channels, layer)
            in_channels = layer.channels

    def add_layer(self, cfg, in_channels, layer):
        conv_layer = nn.Conv2d(in_channels, layer.channels, kernel_size=layer.kernel_size, padding=1).to(self.device)
        nn_ops.initialize_weights(conv_layer, cfg.model.weights_init, cfg.seed)
        
        nonlinearity = nn_ops.get_nonlinearity(cfg.model.nonlin)
        normalization = nn_ops.get_normalization(layer.channels, cfg.model.norm)
        pooling = nn_ops.get_pooling(layer.pooling, layer.pool_kernel_size)

        layer_sequence = nn.Sequential(conv_layer, nonlinearity, normalization, pooling)
        self.conv_layers.append(layer_sequence)

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.conv_layers:
            x = layer(x)
        return x
