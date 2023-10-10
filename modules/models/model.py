# neural networks in practice

from torch import nn
from .components import Unet, unet_output_dim
from ..dataset.mydataset import image_size

class Regression(nn.Module):
    def __init__(self, input_dim, layer_dim):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        predict = self.fc2(x)
        return predict


class Advanced_Regression(nn.Module):
    # fixed input for Unet and MLP, need to be adjusted
    def __init__(self):
        super().__init__()
        self.unet = Unet(
            dim=image_size,
            channels=1,  # here 1 as it is greyscale
            dim_mults=(1, 2, 4,))
        self.mlp = Regression(input_dim=unet_output_dim, layer_dim=256)

    def forward(self, x):
        embedding = self.unet(x)
        flat_emb = embedding.view(embedding.size(0), -1)
        predict = self.mlp(flat_emb)
        return predict
