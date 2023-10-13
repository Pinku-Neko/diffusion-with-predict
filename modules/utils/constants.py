'''
constants for whole project \n
FIXME: consider using json config file to store (part of) these
'''

import torch
import torch.nn.functional as F
from modules.dataset.mydataset import dataset

# total amount of time steps in diffusion
timesteps = 200

# determine which device we run
default_device = "cuda" if torch.cuda.is_available() else "cpu"

# batch size of data loader
default_batch_size = 100

# training epochs
default_training_epochs = 1000

# learning rate
default_learning_rate = 1e-4

# training tolerance for early stopping
default_training_tolerance = 10

# image size, an int value
# selection for finding image_size. Assume all images are same size and square
# FIXME: A better way to find out image_size?
image_size = dataset['train']['image'][0].size[0]

def linear_beta_schedule(timesteps):
    '''
    variance schedule \n
    goal is that each is close to zero, but cummulative product is close to 1
    '''
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps).to(device=default_device)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
