import numpy as np
import torch
from PIL import Image
from torch import Tensor

from consts import CLASSES

model = torch.jit.load('model_scripted_cpu.pt')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(data, target_device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, target_device) for x in data]
    return data.to(target_device, non_blocking=True)


def predict_image(img: Tensor) -> str:
    """Converts image to array and return the predicted class
        with the highest probability"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with the highest probability
    _, predictions = torch.max(yb, dim=1)
    # Retrieve the class label

    return CLASSES[predictions[0].item()]


if __name__ == '__main__':
    image = Image.open('android_image.jpg').resize((256, 256))
    image.show()
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    print(image.shape, image.dtype, torch.randn(3, 256, 256).shape)

    print(predict_image(image))
