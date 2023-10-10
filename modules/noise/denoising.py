# denoise image with time step t to t-1 given model

import torch
from ..utils.helper import extract
from ..utils.constants import timesteps

# tolerance for fast p sample
tolerance = 5

# denoise image with time step t to t-1 given model


def p_sample(model, x, t, t_index):
    from ..utils.constants import betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas, posterior_variance
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

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# experimental recursive method to skip iteration step in p_sample
def fast_p_sample(diffusion, regression, x_t, t, failure):
    # models in eval mode
    diffusion.eval()
    regression.eval()

    # TODO: check the device of model and input

    with torch.no_grad():
        # predict of image in unet and regression
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

    elif (failure <= tolerance):
        # case: predict larger equal than true t, do again
        result = fast_p_sample(diffusion, regression, x_t, t, failure+1)

    else:
        # case: if never goes down, tolerance too high, reduce t
        print(f"failed {tolerance} times at timestep {t}, force down")
        result = p_sample(diffusion, x_t, t-1, t.item())
        result = fast_p_sample(diffusion, regression, result, t-1, 0)

    return result
