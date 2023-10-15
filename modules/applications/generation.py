'''
generate images using normal p_sample and with predict
'''

from tqdm.auto import tqdm
from modules.images.tensors import random_noise
from modules.models.tools import predict
from ..utils import constants as const
from ..utils.helper import exists
from ..noise.denoising import p_sample
from ..images.transforms import reverse_transform

def double_generation(diffusion, regression):
    '''
    generate 2 lists of images using diffusion from same noise \n
    -diffusion: diffusion model \n
    -regression: regression model to predict \n
    -return: images generated from begin and from predict respectively
    '''
    # make 2 images with same noise
    image_shape = (const.image_size,const.image_size)
    noise = random_noise(image_shape)
    images_normal = generate_animation(diffusion,noise=noise)
    images_predict = generate_animation(diffusion,regression,noise)

    # return 2 animations
    return images_normal, images_predict

def generate_animation(diffusion, regression=None, noise = None):
    '''
    generate a list of images generated from diffusion model \n
    -diffusion: diffusion model \n
    -regression: regression model to predict \n
    -noise: start noise tensor \n
    -return: fully generated list of images if regression given, otherwise images generated with predict
    '''
    # generate pure noise
    if noise is None:
        image_shape = (const.image_size,const.image_size)
        noise = random_noise(image_shape)
    
    # use regression to predict if regression exists, otherwise use last t
    t = const.timesteps-1
    timestep = predict(noise, t, regression)
    
    images = []
    # loop p sample
    for i in tqdm(reversed(range(timestep.item()+1)),desc="loop p sampling",total=timestep.item()+1):
        noise = p_sample(diffusion,noise,timestep)
        timestep -= 1
        image = reverse_transform(noise.squeeze(0).to('cpu'))
        images.append(image)

    return images

def generate_image(diffusion, regression=None):
    '''
    generate an image generated from diffusion model \n
    -diffusion: diffusion model \n
    -regression: regression model to predict \n
    -return: fully generated image if regression given, otherwise image generated with predict
    '''
    images = generate_animation(diffusion, regression=None)
    return images[-1]

