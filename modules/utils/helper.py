'''
helper functions for other modules
'''
import time

# Used for forward and reverse process

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