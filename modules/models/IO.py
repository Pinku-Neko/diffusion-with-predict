'''
use to manipulate models on disk
'''

from torch import save, load

def save_model(model, lr, layer_dim=None):
    '''
    save a model with its epoch number and loss as pickle file \n
    model: the model trained \n
    epoch_number: the epoch it is trained \n
    loss: the loss calculated \n
    '''
    # overwrite instead
    # filename = f'./saved_models/regression_epoch_{epoch_number}_loss_{rounded_loss}.pth'
    if layer_dim is not None:
        filename = f'./saved_models/regression_best_{layer_dim}_{lr}.pt'
    else:
        filename = f"./saved_models/diffusion_best_{lr}.pt"
    save(model.state_dict(), filename)


def load_model(model, filename):
    '''
    read a file from disk \n
    model: the model trained. Required to identify the structure of model \n
    filename: the name of file in disk \n
    return: model
    '''
    # Load the model weights
    checkpoint = load(filename)
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {filename}")
    return model