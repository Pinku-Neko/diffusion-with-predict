'''
custom loss function
'''
from torch import nn

class weighted_MSE_loss(nn.Module):
    def __init__(self, weights):
        '''
        weighted MSE loss \n
        -weights: a np array. Normalized regarding length is optimal
        '''
        super(weighted_MSE_loss, self).__init__()
        self.weights = weights

    def forward(self, predicted, target):
        '''
        return weighted MSE loss \n
        the weight is selected based on target value
        '''
        mse_loss = nn.MSELoss()(predicted, target)
        
        weight = self.weights[target]
        weighted_mse_loss = mse_loss * weight

        return weighted_mse_loss
