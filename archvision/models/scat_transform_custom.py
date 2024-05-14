import torch.nn as nn
import torch
import archvision.models.nn_ops as nn_ops
from archvision.models.custom_operations.scat_transform import ScatTransform



class ScatTransformCustom(nn.Module):

    def __init__(
        self,
        num_classes=200,
        trainable_layers=None,
        nonlinearity="relu",
        dropout=True,
        batchnorm=True,
        pooling_type="max",
    ):
        super(ScatTransformCustom, self).__init__()

        trainable_layers = trainable_layers or {'st':"111111","fc": "001"}
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace = True)

        st_1 = ScatTransform(C = 3, J = 1, L = 4, M = 64, N = 64)
        st_2 = ScatTransform(C = 128, J = 1, L = 8, M = 32, N = 32)
        st_3 = ScatTransform(C = 128, J = 1, L = 8, M = 16, N = 16)
        
        scattransform_layers = [st_1,
                                nn.Flatten(1,2),
                                nn.Conv2d(st_1.channel_size, 128, kernel_size=1),
                                st_2,
                                nn.Flatten(1,2),
                                nn.Conv2d(st_2.channel_size, 128, kernel_size=1),
                                st_3,
                                nn.Flatten(1,2),
                                nn.Conv2d(st_3.channel_size, 128, kernel_size=1)
                               ]
        scattransform_layers = [layer for layer in scattransform_layers if layer is not None]
        self.scattransform_layers = nn.Sequential(*scattransform_layers)


        
        classifier_layers = [
            nn.Dropout() if dropout else None,
            nn.Linear(128 * 8 * 8, 1024),
            nonlin_fn,
            nn.BatchNorm1d(1024) if batchnorm else None,
            nn.Linear(1024, 1024),
            nonlin_fn,
            nn.Dropout() if dropout else None,
            nn.Linear(1024, num_classes),
        ]
        classifier_layers = [layer for layer in classifier_layers if layer is not None]
        self.classifier = nn.Sequential(*classifier_layers)

        st_idx = 0
                
        for module in self.classifier:
            if isinstance(module, nn.Conv2d):
                module.requires_grad_(trainable_layers["st"][fc_idx])
                st_idx += 1
                
        fc_idx = 0
        print("Trainable layers: ", trainable_layers)
                
        for module in self.classifier:
            if isinstance(module, nn.BatchNorm1d):
                module.requires_grad_(True)
            elif isinstance(module, nn.Linear):
                module.requires_grad_(trainable_layers["fc"][fc_idx])
                fc_idx += 1

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor containing the image data.

        Returns:
            torch.Tensor: The output tensor containing the class probabilities.
        """
        x = self.scattransform_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
