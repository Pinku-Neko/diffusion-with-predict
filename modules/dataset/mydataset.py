# generating data set and data loader
from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    '''
    generate a set of images and transformed if function is given
    -images: PIL images
    -transform: composed function using torchvision.transforms
    -return: a set of (transformed) images acessible with index
    '''
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image



