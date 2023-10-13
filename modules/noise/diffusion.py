'''
give arbitrary noisy image given image and noise level t
'''

import torch
from ..utils.helper import extract
from ..utils.constants import sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


def q_sample(x_start, t, noise=None):
    '''
    blur the image tensors with given levels of noise \n
    x_start: image tensors \n
    t: noise levels, each should be a tensor of shape (1) \n
    noise: optional, normally random in process \n
    return: noisy image tensors
    '''
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
