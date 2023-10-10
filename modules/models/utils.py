# model manipulation

import re
from torch import save, load

from .components import Unet,unet_output_dim
from .model import Advanced_Regression
from ..dataset.mydataset import image_size
from ..utils.constants import default_device


def save_model(model, epoch_number, loss):
    rounded_loss = round(loss, 6)
    filename = f'./saved_models/regression_epoch_{epoch_number}_loss_{rounded_loss}.pth'
    save(model.state_dict(), filename)
    print(f'save as {filename}')


def load_model(model, filename):
    # Use regular expressions to extract the epoch number from the filename
    match = re.search(r'epoch_(\d+)_loss_([\d.]+)\.pth', filename)

    if match:
        epoch_number = int(match.group(1))
        loss = float(match.group(2))
        # Load the model weights
        checkpoint = load(filename)
        model.load_state_dict(checkpoint)

        print(f"Loaded model from {filename} with epoch {epoch_number}")
        return model, epoch_number, loss
    else:
        raise ValueError("Filename does not match the expected pattern")


def init_models():
    # init model
    diffusion = Unet(
        dim=image_size,
        channels=1,  # here 1 as it is greyscale
        dim_mults=(1, 2, 4,))

    # Create an instance of the MLP model
    regression = Advanced_Regression()

    # move to device
    diffusion, regression = diffusion.to(
        default_device), regression.to(default_device)

    return diffusion, regression
