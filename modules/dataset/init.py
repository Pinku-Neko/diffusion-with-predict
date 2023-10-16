'''
defines the dataset and loads it in the project
'''

# loading dataset
from datasets import load_dataset

dataset = load_dataset('cifar10')
'''this data set has 50k 32x32 rgb images for train and 10k of same size for testing in 10 classes'''

print("dataset loaded")

# take cats
dataset = dataset.filter(lambda example: example['label'] == 3)

print("cats")

