import torch
from torch import nn

class DivNorm(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    
    def forward(self,x):        

        std = x.std(dim=1, keepdims=True)
        mean = x.mean(dim=1, keepdims=True)
        x_norm = (x - mean)/std
        return x_norm

