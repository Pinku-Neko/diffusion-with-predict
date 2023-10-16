'''
get a random image or by index of dataset
'''
import random
from ..dataset.init import dataset

def get_image(index = None):
    if index is None:
        index = int(random.random()*len(dataset['test']['img']))
    return dataset['test']['img'][index]
    