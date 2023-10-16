'''
custom nn models used in project
'''

from torch import nn
from .components import Unet, unet_output_dim
from ..utils.constants import image_size

class Regression(nn.Module):
    def __init__(self, input_dim, layer_dim):
        '''
        ordinary regression model with given input and layer dim \n
        1 hidden layer \n
        all linear fully connected with relu \n
        -input_dim: dimension of input layer \n
        -layer_dim: dimension of 1 hidden layer \n
        -return: model, which outputs a value
        '''
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        predict = self.fc2(x)
        return predict

class Advanced_Regression(nn.Module):
    '''
    a model consisting of trainable unet and regression \n
    -layer_dim: the dimension of hidden layer in part regression
    -return: a model, which outputs a value
    '''
    # fixed input for Unet and MLP, need to be adjusted
    def __init__(self,layer_dim):
        super().__init__()
        self.unet = Unet(
            dim=image_size,
            channels=3,  # here 3 using rgb
            dim_mults=(1, 2, 4,))
        self.mlp = Regression(input_dim=unet_output_dim, layer_dim=layer_dim)

    def forward(self, x):
        embedding = self.unet(x)
        flat_emb = embedding.view(embedding.size(0), -1)
        predict = self.mlp(flat_emb)
        return predict
