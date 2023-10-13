# generating data set and data loader
from torch.utils.data import Dataset, DataLoader
from .mydataset import dataset

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



def prepare_dataset(batch_size, transform):
    '''
    use the predefined dataset to generate shuffled data loader with given batch size
    -precondition: the data set has 'train' and 'test' labels, whose labels are 'label' and 'image'
    -precondition: a transform from PIL to tensor is passed
    -batch_size: maximal size of each batch 
    -return: train- and data loader with size maximal batch_size of each batch, shuffled
    '''
    # train data loader
    train_dataset = Image_Dataset(
        images=dataset['train']['image'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test data loader
    test_dataset = Image_Dataset(
        images=dataset['test']['image'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader