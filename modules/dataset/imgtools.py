'''
manipulates images
'''

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Lambda, ToPILImage
from torch import clamp, uint8
from ..noise.denoising import p_sample
from ..utils.constants import image_size


transform = Compose([
    Resize(image_size),

    CenterCrop(image_size),

    ToTensor(), # turn into Numpy array of shape HWC, divide by 255

    Lambda(lambda t: (t * 2) - 1), # shift [0,1] to [-1,1]
])

# transform from distributions to image
reverse_transform = Compose([
    Lambda(lambda t: clamp(t, min=-1., max=1.)), # clamp [-1,1]

    Lambda(lambda t: (t + 1.) / 2.), # from [-1,1] to [0,1]

    Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC

    Lambda(lambda t: t * 255.), # from [0,1] to [0,255]

    Lambda(lambda t: t.astype(uint8)), # round to unsgined int8
    
    ToPILImage(),
])

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
    x_noisy = p_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy)

    return noisy_image