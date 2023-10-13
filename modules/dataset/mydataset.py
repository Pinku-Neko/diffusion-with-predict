'''
defines the dataset and loads it in the project \n
FIXME: first time loading will raise error because of circular dependencies
'''

# loading dataset
from datasets import load_dataset

dataset = load_dataset('fashion_mnist')
'''this data set has 60k 28x28 greyscale images for train and 10k of same size for testing'''


print("dataset loaded")





