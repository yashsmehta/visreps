import torch.nn as nn
import torch
import visreps.models.nn_ops as nn_ops
from visreps.models.custom_operations.scat_transform import ScatTransform



class ScatTransformFront(nn.Module):

    def __init__(
        self,
        num_classes=200,
        trainable_layers=None,
        nonlinearity="relu",
        dropout=True,
        batchnorm=True,
        pooling_type="max",
    ):
        super().__init__()

        trainable_layers = trainable_layers or {"fc": "111"}
        trainable_layers = {
            layer_type: [val == "1" for val in layers]
            for layer_type, layers in trainable_layers.items()
        }

        nonlin_fn = nn_ops.get_nonlinearity(nonlinearity, inplace=True)
        self.scattransform = ScatTransform(J=3, L=8, M=64, N=64)

        classifier_layers = [
            nn.Dropout() if dropout else None,
            nn.Linear(self.scattransform.layer_size, 1024),
            nonlin_fn,
            nn.BatchNorm1d(1024) if batchnorm else None,
            nn.Linear(1024, 1024),
            nonlin_fn,
            nn.Dropout() if dropout else None,
            nn.Linear(1024, num_classes),
        ]
        classifier_layers = [layer for layer in classifier_layers if layer is not None]
        self.classifier = nn.Sequential(*classifier_layers)
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
        x = self.scattransform(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
