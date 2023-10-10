# helper functions
import time

# Used for forward and reverse process

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a[t]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def record_time(message, prev_time = None):
    if prev_time is None:
        prev_time = time.time()
    curr_time = time.time()
    diff = round(curr_time - prev_time,ndigits=8)
    print(f"{message}: {diff}s")
    return curr_time