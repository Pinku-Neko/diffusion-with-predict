'''
verify the ability of model using plots
'''

import numpy as np
import random
from matplotlib import pyplot as plt
import torch
from tqdm.auto import tqdm
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
    regression.eval()
    num_samples = 128

    # indices for image samples
    indices = random.sample(range(0,10000), num_samples)

    # transform into distributions
    image_tensors = torch.stack([transform(dataset['test']['image'][index]) for index in tqdm(indices)],dim=0)

    # results as nd array
    # 2nd axis: average and std_dev
    result = np.zeros((timesteps,2))

    for i in tqdm(range(timesteps)):
        # time
        time = torch.tensor([i]).repeat(num_samples)

        # calculate noise
        noise = q_sample(image_tensors.to(default_device),time)
        
        # pass to model
        with torch.no_grad():
            predict = regression(noise)
        predict = (predict * timesteps).squeeze().detach().to('cpu')

        # store difference
        difference = torch.abs(predict - time)

        # calculate average
        average = torch.mean(difference).item()
        
        # calculate standard deviation
        std_dev = torch.std(difference).item()

        result[i] = np.array([average, std_dev])

    # draw average
    x_values = range(timesteps)
    plt.plot(x_values, result[:,0], label='Average', color='black')
    plt.errorbar(x = x_values, y = result[:,0],yerr=result[:,1],label = 'Standard Deviation', fmt='o', markersize=1, color = 'blue')

    # plot
    plt.xlabel('Timestep')
    plt.ylabel('Error in timestep')
    plt.title(f'Error of prediction from true timestep. Sample size:{num_samples}')
    plt.legend()
    plt.grid(True)
    plt.show()