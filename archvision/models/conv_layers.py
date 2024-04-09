import torch.nn as nn
import os
from models.custom_operations.convolution import WaveletConvolution
from models.custom_operations.norm import DivNorm

class ConvolutionLayers(nn.Module):
    def __init__(self, cfg):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        # in_channels = cfg.in_channels
        in_channels = 3

        for i, out_channels in enumerate(cfg.conv.layer_sizes):
            
            if out_channels == None: #create conv layer with fixed engneered filters
                conv_layer = WaveletConvolution(filter_size=cfg.conv.kernel_sizes[i], 
                                                filter_type=cfg.conv.kernel_types[i])
                out_channels = conv_layer.layer_size
                
                
            else:
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
                case 'batch':
                    norm_layer = nn.BatchNorm2d(out_channels)    
                case 'channel':
                    norm_layer = DivNorm()
                # case "layer":
                #     norm_layer = nn.LayerNorm(out_channels)
                case "instance":
                    norm_layer = nn.InstanceNorm2d(out_channels)
                # case "group":
                #     norm_layer = nn.GroupNorm(num_groups=32, num_channels=out_channels)
                # case "weight":
                #     norm_layer = nn.utils.weight_norm(conv_layer)
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

            
            match cfg.conv.pool_type:
                case 'max':
                    if cfg.conv.pool_kernel_sizes[i] == None:
                        pool_layer = nn.Identity()
                    else:
                        pool_layer = nn.MaxPool2d(kernel_size=cfg.conv.pool_kernel_sizes[i])
                
                case 'avg':
                    if cfg.conv.pool_kernel_sizes[i] == None:
                        pool_layer = nn.Identity()
                    else:
                        pool_layer = nn.AvgPool2d(kernel_size=cfg.conv.pool_kernel_sizes[i])

                case _:
                    raise ValueError(f"Unsupported pool type: {cfg.conv.pool_type}")
            
            
            self.conv_layers.append(nn.Sequential(conv_layer, norm_layer, nonlin_layer, pool_layer))
            in_channels = out_channels

    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x











    