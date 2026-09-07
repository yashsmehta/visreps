"""NSD-synthetic input transform matching the dataset authors' reference code."""
import numpy as np
from PIL import Image
from torchvision import transforms

class SyntheticReferenceTransform:
    def __init__(self):
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])

    def __call__(self, image):
        image = Image.fromarray((np.sqrt(np.asarray(image.convert('RGB')) / 255) * 255).astype(np.uint8))
        image = transforms.CenterCrop(min(image.size))(image)
        return self.normalize(self.to_tensor(self.resize(image)))

    def __repr__(self):
        return 'SyntheticReferenceTransform(sqrt_uint8, CenterCrop(min_size), Resize(224,224,bilinear), ToTensor, ImageNetNormalize)'
