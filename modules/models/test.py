'''
verify the ability of model using plots
'''

from numpy import zeros
import random
from matplotlib import pyplot as plt
from torch import tensor, arange
from ..utils.constants import timesteps, default_device
from ..dataset.mydataset import dataset
from ..dataset.imgtools import transform
from ..noise.diffusion import q_sample

def evaluate_regression(regression):
    '''
    evaluate the regression model by plotting samples with predict error \n
    1. draw n samples from the dataset \n
    2. make timesteps * n array \n
    3. for each
    '''
    # draw 3 samples, each from 0 to 199
    # make 3 inputs
    # make 200 * 3 array
    # for each, do q_sample
    # predict t and compare with true t
    # store difference
    num_samples = 20

    # indices for image samples
    indices = random.sample(range(0,60000), num_samples)

    # transform into distributions
    image_tensors = [transform(dataset['train']['image'][index]) for index in indices]

    # results as nd array
    result = zeros((timesteps,num_samples))

    for i in range(num_samples):
        # gather input distribution
        input = image_tensors[i].repeat(timesteps,1,1)

        # time
        time = arange(start=0, end=timesteps)

        # calculate noise
        noise = q_sample(input.to(default_device),time)
        
        # pass to model
        predict = regression(noise) * timesteps

        # store difference
        result[:,i] = predict - time

    # draw samples
    for i in range(num_samples):
        plt.plot(range(timesteps), result[:,i], label=f'Sample {i+1}, index {indices[i]}')

    # plot
    plt.xlabel('Timestep')
    plt.ylabel('Error in timestep')
    plt.title('Error of prediction from true timestep')
    plt.legend()
    plt.grid(True)
    plt.show()