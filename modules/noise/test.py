from matplotlib import pyplot as plt
from ..dataset.mydataset import train_images
from ..dataset.imgtools import transform, reverse_transform
from ..utils.constants import default_device
from ..noise.denoising import fast_p_sample

def fast_sample_test(image_index, diffusion, regression, noise_strength):
    '''
    test doc \n
    line 2
    '''
    # make image
    input = transform(train_images[image_index])

    # convert value to tensor for sample
    t = noise_strength.to(default_device).view(1)

    # output distribution
    output = fast_p_sample(diffusion,regression,x_t=input,t=t,tolerance=0)

    # transform back to image
    # TODO: test functionality and look how to improve the readability
    result = reverse_transform(output.squeeze(dim=0).detach().to('cpu'))
    # result = reverse_transform(output.squeeze(dim=0).detach().to('cpu'))

    title = f"index: {image_index}; noise: {noise_strength}"
    plt.imshow(result,cmap='gray')
    plt.title(title)
    plt.show()