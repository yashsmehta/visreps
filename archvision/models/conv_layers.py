import torch.nn as nn


class ConvolutionLayers(nn.Module):
    def __init__(self, cfg):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        # in_channels = cfg.in_channels
        in_channels = 3

        for i, out_channels in enumerate(cfg.conv.layer_sizes):
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=cfg.conv.kernel_sizes[i], padding=1)

            match cfg.conv.init:
                case "xavier":
                    nn.init.xavier_normal_(conv_layer.weight)
                case "kaiming":
                    nn.init.kaiming_normal_(conv_layer.weight)
                case "gaussian":
                    nn.init.normal_(conv_layer.weight, mean=0, std=0.02)
                case "uniform":
                    nn.init.uniform_(conv_layer.weight, a=-0.02, b=0.02)
                case _:
                    raise ValueError(f"Unsupported initialization method: {cfg.conv.init}")

            match cfg.conv.norm:
                case "layer":
                    norm_layer = nn.LayerNorm(out_channels)
                case "instance":
                    norm_layer = nn.InstanceNorm2d(out_channels)
                case "group":
                    norm_layer = nn.GroupNorm(num_groups=32, num_channels=out_channels)
                case "weight":
                    norm_layer = nn.utils.weight_norm(conv_layer)
                case "none":
                    norm_layer = nn.Identity()
                case _:
                    raise ValueError(f"Unsupported normalization method: {cfg.conv.norm}")

            match cfg.conv.nonlin:
                case "relu":
                    nonlin_layer = nn.ReLU(inplace=True)
                case "tanh":
                    nonlin_layer = nn.Tanh()
                case "sigmoid":
                    nonlin_layer = nn.Sigmoid()
                case "elu":
                    nonlin_layer = nn.ELU(inplace=True)
                case "none":
                    nonlin_layer = nn.Identity()
                case _:
                    raise ValueError(f"Unsupported non-linearity: {cfg.conv.nonlin}")

            self.conv_layers.append(nn.Sequential(conv_layer, norm_layer, nonlin_layer))
            in_channels = out_channels

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x