'''
get noisy image
'''

from torch import tensor
from ..utils.constants import default_device,timesteps
from ..images.transforms import reverse_transform
from ..images.transforms import transform
from ..noise.diffusion import q_sample


def animate_noisy_image(input_image, time=None):
    images = []
    if time is None:
        time = timesteps
    for i, step in enumerate(range(time)):
        result_image = get_noisy_image(input_image,i)
        images.append(result_image)
    return images

def get_noisy_image(image, t):
    '''
    adds gaussian noise given level to an image \n
    -image: PIL image \n
    -t: int value \n
    -return: noisy image in PIL
    '''
    # transform to distributions
    x_start = transform(img=image).to(default_device)

    # add noise
    t = tensor([t]).to(default_device)
    x_noisy = q_sample(x_start, t=t).to('cpu')

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy)

    return noisy_image