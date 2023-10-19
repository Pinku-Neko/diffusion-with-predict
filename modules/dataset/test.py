'''
contains the dataset for testing the model
'''

from torch import tensor,arange
from torch.utils.data import Dataset
from ..utils.constants import timesteps,default_device

class Single_Image_Dataset(Dataset):
    '''
    use a single image to generate a dataset with all noise levels
    -transform: a transform function to convert image into tensor 
    -image: a PIL image
    -return: a dataset of image tensors with timesteps many noise levels 
    '''
    def __init__(self, image, transform):
        # all images require a transform
        image = transform(img=image).to(default_device)
        
        # 2 dimensions for dimensions of 2-D image tensor
        self.images = image.repeat(timesteps,1,1)

        # make each image tensor noisy from 0 to timesteps-1
        from modules.noise.diffusion import q_sample
        self.images = q_sample(self.images,arange(start=0,end=timesteps,device=default_device))

    def __len__(self):
        # not access timesteps in case bugs occur 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


class Diff_Single_Image_Dataset(Dataset):
    '''
    use a single image to generate a dataset with all noise levels
    -transform: a transform function to convert image into tensor 
    -image: a PIL image
    -return: a dataset of image tensors with timesteps many noise levels 
    '''
    def __init__(self, image, transform):
        # all images require a transform
        image = transform(img=image).to(default_device)
        
        # 2 dimensions for dimensions of 2-D image tensor
        self.images = image.repeat(timesteps,1,1,1)

    def __len__(self):
        # not access timesteps in case bugs occur 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image