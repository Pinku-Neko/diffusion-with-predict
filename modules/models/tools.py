'''
handles model manipulation
'''

# init models
from modules.utils import constants as const
import torch
from .components import Unet,unet_output_dim
from .model import Advanced_Regression
from ..utils.constants import image_size,default_device


def init_models(regression_layer_dim):
    '''
    init model
    -regression_layer_dim: layer_dim of linear layer in regression model \n
    -return: blank unet model and unet+mlp regression model, in device
    '''
    diffusion = Unet(
        dim=image_size,
        channels=3,  # here 3 as rgb
        dim_mults=(1, 2, 4,))

    # Create an instance of the MLP model
    regression = Advanced_Regression(layer_dim=regression_layer_dim)

    # move to device
    diffusion, regression = diffusion.to(
        default_device), regression.to(default_device)

    return diffusion, regression


def predict(noise, regression=None, true_t=None):
    '''
    predict the timestep based on noise input \n
    -noise: start noise \n
    -regression: regression model for predit \n
    -return: timestep of predict if regression is given, otherwise last element of timesteps
    '''
    if true_t is None:
        t = const.timesteps-1
    else:
        t = true_t
    timestep = torch.tensor([t]).to(const.default_device)
    if regression is not None:
        regression.eval()
        with torch.no_grad():
            timestep = (regression(noise).squeeze(0) * const.timesteps).to(torch.int)
    return timestep
