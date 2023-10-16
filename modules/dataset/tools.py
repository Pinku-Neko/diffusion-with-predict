from modules.dataset.init import dataset
from modules.dataset.mydataset import Image_Dataset


from torch.utils.data import DataLoader


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
        images=dataset['train']['img'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test data loader
    test_dataset = Image_Dataset(
        images=dataset['test']['img'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader