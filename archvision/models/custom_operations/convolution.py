from models.custom_operations.preset_filters import filters, generate_discrete_wavelet_family
from torch.nn import functional as F
import math
import torch
torch.manual_seed(42)
from torch import nn
import numpy as np
import pywt

discrete_wavelets = ['bior', 'coif', 'db', 'dmey', 'haar', 'rbio', 'sym']
cont_wavelets = ['cmor' , 'shan' , 'fbsp' ,'cgau' ,'gaus' ,'mexh' ,'morl']


class WaveletConvolution(nn.Module):
    
    
    def __init__(self, 
                 filter_type:str,
                 filter_params:dict=None,
                 filter_size:int=None,
                 device:str=None

                ):
                
        super().__init__()
        

        self.filter_type = filter_type
        self.filter_size = filter_size
        self.filter_params = get_kernel_params(self.filter_type)
        self.layer_size = get_layer_size(self.filter_type, self.filter_params)
        self.device = device
        
    
    def forward(self,x):
            
        x = x.to(self.device)
        
        in_channels = x.shape[1]
        
        convolved_tensor = []
        
        if self.filter_type in ['curvature','gabor']:
            weights = filters(in_channels=1, kernel_size=self.filter_size, filter_type = self.filter_type, filter_params=self.filter_params).to(self.device)
            for i in range(in_channels):
                    channel_image = x[:, i:i+1, :, :]
                    channel_convolved = F.conv2d(channel_image, weight= weights.to(self.device), padding=weights.shape[-1] // 2 - 1)
                    convolved_tensor.append(channel_convolved)
                    
        elif self.filter_type in discrete_wavelets:
            weights = generate_discrete_wavelet_family(wavelet_family=self.filter_type)
            for w in weights:
                for i in range(in_channels):
                    channel_image = x[:, i:i+1, :, :]
                    channel_convolved = F.conv2d(channel_image, weight= w.to(self.device), padding=w.shape[-1] // 2 - 1)
                    convolved_tensor.append(channel_convolved)

        elif self.filter_type in cont_wavelets:
            weights = generate_continuous_wavelet_filters(wavelet_family=self.filter_type, num_scales=3)
            
            for w in weights:
                for i in range(in_channels):
                    channel_image = x[:, i:i+1, :, :]
                    channel_convolved = F.conv2d(channel_image, weight= w.to(self.device), padding=w.shape[-1] // 2 - 1)
                    convolved_tensor.append(channel_convolved)
            
            
        
        # for RGB input (the preset L1 filters are repeated across the 3 channels)
        
        x = torch.cat(convolved_tensor, dim=1)   
 
        return x    




def get_kernel_params(kernel_type):
    
    if kernel_type == 'curvature':
        return {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]}
    
    elif kernel_type == 'gabor':
         return {'n_ories':12,'num_scales':3}
        
    elif kernel_type in discrete_wavelets:
        return None

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")



def get_layer_size(kernel_type, kernel_params):


        if kernel_type == 'curvature':
            return kernel_params['n_ories']*kernel_params['n_curves']*len(kernel_params['gau_sizes']*len(kernel_params['spatial_fre']))*3
        
        elif kernel_type == 'gabor':
            return kernel_params['n_ories']*kernel_params['num_scales']*3
       
        elif kernel_type in discrete_wavelets:
            wavelet_list = [i for i in pywt.wavelist() if kernel_type in i] 
            return len(wavelet_list) * 2 * 3
        
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")