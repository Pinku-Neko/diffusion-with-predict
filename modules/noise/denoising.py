'''
denoise image tensors to less noisy image tensors
'''

import torch
from ..utils.helper import extract
from ..utils.constants import timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas, posterior_variance

# tolerance for fast p sample
default_tolerance = 5

def p_sample(model, x, t):
    '''
    denoise image tensor with time step t to t-1 given diffusion model \n
    -precondition: model, x and t are on the same device \n
    -model: diffusion model, unet \n
    -x: image tensors as input of model \n
    -t: noise level, each element a tensor shape (1) as input of model \n
    -return: an image tensor with noise level t-1
    '''
    # switch to eval mode (regarding deactivate batch norm etc.)
    model.eval()

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    with torch.no_grad():
        predict_noise = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predict_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t.item() == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise



def fast_p_sample(diffusion, regression, x_t, t, failure):
    '''
    experimental recursive method to skip iteration step in p_sample \n
    -precondition: 2 models, and x_t and t are on the same device \n
    -diffusion: model for denoising \n
    -regression: model for predicting \n
    -x_t: image tensor \n
    -t: noise level as tensor shape (1) \n
    -failure: how many failed attempts this far \n
    -if predict lower than true value: return denoised tensor with predict noise level \n
    -if predict larger equal than true value: return method recursively with failure+1 \n
    -if failure too high: return denoised tensor with true noise level
    '''
    # models in eval mode
    diffusion.eval()
    regression.eval()

    with torch.no_grad():
        # predict of image tensor in regression
        next_t = regression(x_t) * timesteps

    # convert into appropriate value
    # here assummed value < 256
    next_t = next_t.to(torch.int8).view(1)

    # clamp between range of timesteps
    next_t = torch.clamp(next_t, min=0, max=timesteps-1)

    # check whether the predict proceeds
    if next_t >= timesteps and next_t < 0:
        # value error
        raise ValueError(f"invalid predict time ]0,{timesteps-1}[")

    if (next_t == 0):
        # end of recursion, return last step
        result = p_sample(diffusion, x_t, next_t, next_t.item())
        pass

    elif (next_t < t):
        # case: predict smaller than true t. return p_sample, keep going
        print(f"from t {t} to predict {next_t}")
        result = fast_p_sample(diffusion, regression, result, next_t, failure)

    elif (failure < default_tolerance):
        # case: predict larger equal than true t, do again
        result = fast_p_sample(diffusion, regression, x_t, t, failure+1)

    else:
        # case: if never goes down, failure too high, reduce t
        print(f"failed {default_tolerance} times at timestep {t}, force down")
        result = p_sample(diffusion, x_t, t-1, t.item())
        result = fast_p_sample(diffusion, regression, result, t-1, 0)

    return result
