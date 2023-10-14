'''
handles model manipulation
'''

# init models
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
        channels=1,  # here 1 as it is greyscale
        dim_mults=(1, 2, 4,))

    # Create an instance of the MLP model
    regression = Advanced_Regression(layer_dim=regression_layer_dim)

    # move to device
    diffusion, regression = diffusion.to(
        default_device), regression.to(default_device)

    return diffusion, regression
