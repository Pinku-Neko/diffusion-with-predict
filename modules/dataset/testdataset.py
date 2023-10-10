# used for testing power of model
from datasets import Dataset
from torch import range
from ..noise.diffusion import q_sample

# used for testing
class Single_Image_Dataset(Dataset):
    def __init__(self, image, num_samples):
        self.num_samples = num_samples
        # make each image noisy
        # 2 1s for fix the dimension of image
        # FIXME: t is inappropriate 
        self.images = q_sample(image.repeat(num_samples, 1, 1), range(0,num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        return image
