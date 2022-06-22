import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from datasetcreator import DatasetCreator
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

PATH = "?"

data = DatasetCreator.images(PATH, 100, 100)

data = data.astype('float32') # Necessary. Otherwise the normalization will return only black squares

#data = data/127.5 - 1.0 # Dangerous operation --> Might collapse CPU

for i in range(len(data)):
    data[i] = data[i]/127.5 - 1.0 # This one probably does the same, but in a smoother way. It won't collapse your CPU...I guess...



    
sample = data * 0.2 # Sampling 20% of the data as X_train
X_train = data[:sample]

classes = ("Normal", "Threat")

for i in range(len(X_train)): # Plotting the images so we can label them
    X_train[i] = (X_train[i]+1.0)*0.5 # Denormalizing
    plt.imshow(X_train[i])
    plt.title(f"image {i}")
    plt.show()

y_train = [1, 1, 0, 0, 2, 3, 2, 1] # Continue on as you wish.

y_train = np.array(y_train)

print(y_train[0])
print(y_train.shape)

print(len(X_train), len(y_train))

X_test = data[sample:sample+50]

for i in range(X_test.shape[0]):
    X_test[i] = (X_test[i]+1.0)*0.5
    plt.imshow(X_test[i])
    plt.title(f'image {i}')
    plt.show()
    
y_test = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
        1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0,
        1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 0]

y_test = np.array(y_test)

print(y_test[0])
print(y_test.shape)
