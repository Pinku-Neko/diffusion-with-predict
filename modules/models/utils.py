'''
handles model manipulation
'''

# reading file, IO disk
import re
from torch import save, load

# init models
from .components import Unet,unet_output_dim
from .model import Advanced_Regression
from ..utils.constants import image_size,default_device


def save_model(model, layer_dim, lr):
    '''
    save a model with its epoch number and loss as pickle file \n
    model: the model trained \n
    epoch_number: the epoch it is trained \n
    loss: the loss calculated \n
    '''
    # overwrite instead
    # filename = f'./saved_models/regression_epoch_{epoch_number}_loss_{rounded_loss}.pth'
    filename = f'./saved_models/regression_best_{layer_dim}_{lr}.pth'
    save(model.state_dict(), filename)


def load_model(model, filename):
    '''
    read a file from disk \n
    model: the model trained. Required to identify the structure of model \n
    filename: the name of file in disk \n
    return: model
    '''
    # Load the model weights
    checkpoint = load(filename)
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {filename}")
    return model


def init_models(regression_layer_dim):
    '''
    init model
    -regression_layer_dim: layer_dim of linear layer in regression model \n
    -return: blank unet model and unet+mlp regression model, in device
    '''
    diffusion = Unet(
        dim=image_size,
        channels=1,  # here 1 as it is greyscale
        dim_mults=(1, 2, 4,))

    # Create an instance of the MLP model
    regression = Advanced_Regression(layer_dim=regression_layer_dim)

    # move to device
    diffusion, regression = diffusion.to(
        default_device), regression.to(default_device)

    return diffusion, regression
