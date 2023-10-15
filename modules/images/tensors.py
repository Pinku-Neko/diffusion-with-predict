from modules.utils import constants as const


import torch


def random_noise(image_shape):
    '''
    generate random noise in [-1,1) in given shape
    '''
    noise = (2 * torch.rand(image_shape) - 1).to(const.default_device).unsqueeze(0)
    return noise