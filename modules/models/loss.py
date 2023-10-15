'''
custom loss function
'''
from torch import nn,tensor,int,sum,sqrt
from ..utils.constants import default_device, timesteps

class weighted_MSE_loss(nn.Module):
    def __init__(self, weights):
        '''
        weighted MSE loss \n
        -weights: a np array. Normalized regarding length is optimal
        '''
        super(weighted_MSE_loss, self).__init__()
        self.weights = tensor(weights).to(default_device)

    def forward(self, predicted, target):
        '''
        return weighted MSE loss \n
        the weight is selected based on target value
        '''
        # find out integer value of t
        target_int = (target * timesteps).to(int)

        # assign weights
        weights = self.weights[target_int]

        # multiply sqrt weights with inputs, so square error is effectively multiplied by weights
        weighted_predicted, weighted_target = sqrt(weights)*predicted, sqrt(weights)*target

        # the following 2 return same result, and it is good
        # weighted_mse_loss = sum(weights*((predicted-target)**2))/len(predicted)
        weighted_mse_loss = nn.MSELoss()(weighted_predicted, weighted_target)
        
        return weighted_mse_loss
