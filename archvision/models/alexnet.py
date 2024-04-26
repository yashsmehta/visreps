import sys
import torchvision
import torch
from torch import nn
import pickle
import os
from .last_layer import Last
torch.manual_seed(0)
torch.cuda.manual_seed(0)



class Model(nn.Module):
    def __init__(self, base_model, layer_name, last):
        super(Model, self).__init__()
        self.layer_name = layer_name
        self.base_model = base_model
        self.last = last
        self.activations = {}

        # Register hook to capture the activations
        layer = dict([*self.base_model.named_modules()])[layer_name]
        layer.register_forward_hook(self.get_activation(layer_name))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def get_layers(self):
        return dict([*self.base_model.named_modules()])

    
    def forward(self, x):
        x = self.base_model(x)  # This will trigger the hook
        x = self.activations.get(self.layer_name)  # Retrieve activation
        x = self.last(x)
        return x






class AlexNet:
    def __init__(self, layer_name='features.12'):
        self.layer_name = layer_name

    def build(self):
        
        base_model = torchvision.models.alexnet(pretrained=True)
        last = Last()
        
        return Model(base_model, self.layer_name, last)


