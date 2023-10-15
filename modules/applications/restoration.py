'''
restore noisy image based on models
'''

from ..utils.constants import default_device
from ..images.transforms import transform
from . import generation as gen

def double_restore(image, diffusion, regression = None):
    # image to tensors in device
    image_tensor = transform(image).to(default_device)

    anim_normal, anim_predict = gen.double_generate(diffusion,regression,noise=image_tensor)
    return anim_normal, anim_predict

def restore_animation(image, diffusion, regression = None):
    # image to tensors in device
    image_tensor = transform(image).to(default_device)

    # generate image using tensor
    result_animation = gen.generate_animation(diffusion,regression,image_tensor)
    return result_animation

def restore_image(image, diffusion, regression = None):
    # image to tensors in device
    image_tensor = transform(image).to(default_device)

    # generate image using tensor
    result_image = gen.generate_image(diffusion,regression,image_tensor)
    return restore_image
