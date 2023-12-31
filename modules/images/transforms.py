from numpy import uint8
from torch import clamp
from modules.utils.constants import image_size


from torchvision.transforms import CenterCrop, Compose, Lambda, Resize, ToPILImage, ToTensor


transform = Compose([
    Resize(image_size),

    CenterCrop(image_size),

    ToTensor(),  # turn into Numpy array of shape HWC, divide by 255

    Lambda(lambda t: (t * 2) - 1),  # shift [0,1] to [-1,1]
])

# transform from distributions to image
reverse_transform = Compose([
    Lambda(lambda t: clamp(t, min=-1., max=1.)),  # clamp [-1,1]

    Lambda(lambda t: (t + 1.) / 2.),  # from [-1,1] to [0,1]

    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC

    Lambda(lambda t: t * 255.),  # from [0,1] to [0,255]

    Lambda(lambda t: t.numpy().astype(uint8)),  # round to unsgined int8

    ToPILImage(),
])
