# define the dataset used for the project

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

dataset = load_dataset('fashion_mnist')  # 60k 28x28 greyscale
print("dataset loaded")
train_images = dataset['train']['image']  # selection

# image size, an int value
image_size = train_images[0].size[0]  # assume all images are same size and square

class Image_Dataset(Dataset):
    from .imgtools import transform
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


def prepare_dataset(batch_size):
    from .imgtools import transform
    train_dataset = Image_Dataset(
        images=dataset['train']['image'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = Image_Dataset(
        images=dataset['test']['image'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


from torch import tensor
from modules.noise.diffusion import q_sample

# used for testing
class Single_Image_Dataset(Dataset):
    def __init__(self, image, num_samples):
        self.images = image.repeat(200,1,1) # 2 1s for fix the dimension of image
        self.num_samples = num_samples
        # make each image noisy
        for idx in range(num_samples):
            self.images[idx] = q_sample(image,tensor(idx).view(1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        return image

