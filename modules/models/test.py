from numpy import zeros
from matplotlib import pyplot as plt
from torch import tensor
from ..utils.constants import timesteps, default_device
from ..dataset.mydataset import train_images
from ..dataset.imgtools import transform
from ..noise.diffusion import q_sample

def evaluate_regression(regression):
    # draw 3 samples, each from 0 to 199
    # make 3 inputs
    # make 200 * 3 array
    # for each, do q_sample
    # predict t and compare with true t
    # store difference
    num_samples = 20

    # indices for image samples
    import random
    indices = random.sample(range(0,60000), num_samples)

    # transform into distributions
    inputs = [transform(train_images[index]) for index in indices]

    # results as nd array
    result = zeros((timesteps,num_samples))

    for t in range(timesteps):
        for i in range(num_samples):
            # gather input distribution
            input = inputs[i]

            # time
            time = tensor([t]).to(default_device)

            # calculate noise
            noise = q_sample(input.to(default_device),time)
            
            # pass to model
            predict = regression(noise) * timesteps

            # store difference
            result[t][i] = predict - t

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