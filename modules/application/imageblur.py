from modules.dataset.imgtools import reverse_transform, transform
from modules.noise.diffusion import q_sample


def get_noisy_image(image, t):
    '''
    adds gaussian noise given level to an image \n
    -image: PIL image \n
    -t: tensor shape (1) \n
    -return: noisy image in PIL
    '''
    # transform to distributions
    x_start = transform(img=image)

    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy)

    return noisy_image