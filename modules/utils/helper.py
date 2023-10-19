'''
helper functions for other modules
'''
from inspect import isfunction
import torch
import time
import numpy as np

def extract(a, t, x_shape):
    '''
    pick out the elements in array with given indices from t \n
    x_shape?
    '''
    batch_size = t.shape[0]
    out = a[t]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def record_time(message, prev_time = None):
    '''
    record time and log messages \n
    prev_time: time since last record \n
    -return: current time for next recording
    '''
    if prev_time is None:
        prev_time = time.time()
    curr_time = time.time()
    diff = round(curr_time - prev_time,ndigits=8)
    print(f"{message}: {diff}s")
    return curr_time

def generate_custom_array(value_interval_left, value_interval_right, separate, length, normalize = None):
    '''
    generate a custom array with left having repeated values and right another value \n
    -value_interval_left: value in left interval \n
    -value_interval_right: value in right interval \n
    -separate: where the left value ends \n
    -length: total length of array \n
    -return: [a,...,a,b,...,b] with 'separate' many a's and rest many b's \n
    -if normalize true, then normalized, so that sum is length of array
    '''
    if separate > length:
        raise ValueError("separate index should be less than or equal to length")

    result_array = np.array([value_interval_left] * separate + [value_interval_right] * (length - separate))

    if (normalize):
        result_array = result_array / result_array.sum() * len(result_array)

    return result_array


def exists(x):
    return x is not None


def default(val, d):
    '''
    return val, if val exists \n
    or if d is a function, return result of function \n
    or return d itself
    '''
    if exists(val):
        return val
    return d() if isfunction(d) else d

def normalize(tensor, range=None):
    '''
    normalize tensor [low,high], to [min,max] \n
    default min max is [-1,1] \n
    -tensor: tensor \n
    -range: list as interval [min,max]
    '''
    # lacking case where only one is 
    if range is None:
        min, max = -1, 1
    else:
        min, max = range

    low = torch.min(tensor)
    high = torch.min(tensor)

    # move to 0
    result = tensor - low

    # divide by (high-low)
    result = result / (high-low)

    # multiply with (max-min)
    result = result * (max-min)

    # move to [min,max]
    result = result - min

    return result