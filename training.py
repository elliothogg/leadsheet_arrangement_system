from matplotlib import pyplot
import numpy as np


# example of loading the fashion_mnist dataset
from keras.datasets.fashion_mnist import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

i = 30
# define subplot
# pyplot.subplot(10, 10, 1 + i)
# # turn off axis
# pyplot.axis('on')
# # plot raw pixel data
# pyplot.imshow(trainX[i], cmap='gray_r')
# print(trainy[i])
# pyplot.show()

print(type(load_data()))