import torch.nn as nn
import numpy as np
import pywt
import torch

def create_filters(filter_type, cfg):
    filters = []
    wavelet_families = pywt.families(short=True)
    if filter_type in wavelet_families:
        if filter_type in ['db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp']:
            wavelet = pywt.Wavelet(filter_type)
            filter_tensor = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
            filters.append(filter_tensor)
        elif filter_type == 'haar':
            filter_tensor = torch.tensor([0.7071, 0.7071], dtype=torch.float32)
            filters.append(filter_tensor)
        elif filter_type == 'cmor':
            B = 1.0  # Bandwidth parameter
            C = 1.5  # Center frequency parameter
            wavelet_name = f'cmor{B}-{C}'
            filter_tensor = torch.tensor(pywt.cwt(torch.zeros(1).cpu().numpy(), [1], wavelet=wavelet_name, sampling_period=1)[0].astype(np.float32), dtype=torch.float32)
            filters.append(filter_tensor)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
    else:
        raise ValueError(f"Filter type {filter_type} not found in PyWavelets families: {wavelet_families}")
    print(filter_tensor)
    exit()
    return filters

class WaveletLayers(nn.Module):
    def __init__(self, cfg):
        super(WaveletLayers, self).__init__()
        filter_type = cfg.wavelet.type
        self.layers = nn.ModuleList()
        filter_tensors = create_filters(filter_type, cfg)
        for filter_tensor in filter_tensors:
            kernel_size = filter_tensor.shape[0]
            out_channels = cfg.wavelet.num_orientations * cfg.wavelet.num_scales
            # conv_layer = nn.Conv2d(3, out_channels, kernel_size, padding=kernel_size//2, bias=False)
            # conv_layer.weight.data = filter_tensor.view(1, 3, kernel_size, 1).repeat(out_channels, 1, 1, 1)
            conv_layer = nn.Conv2d(1, out_channels, kernel_size, padding=kernel_size//2, bias=False)
            conv_layer.weight.data = filter_tensor.view(1, 1, kernel_size, 1).repeat(out_channels, 1, 1, 1)
            conv_layer.weight.requires_grad = False
            self.layers.append(conv_layer)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            output = layer(x)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        
        return x