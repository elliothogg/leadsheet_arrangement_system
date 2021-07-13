import pickle
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np
import tensorflow as tf

#prevents scientific number notation
np.set_printoptions(suppress=True)

# load the training data
def load_real_samples():
    with open('training_data_2d.pickle', 'rb') as file:
        training_data_2d = pickle.load(file)
    x2, x1 = training_data_2d
    print(x1.shape)
    print(x2.shape)
    return [x1, x2]


training_data = load_real_samples()

labels = training_data[0]

label = np.array([labels[0]])

model = load_model('model_004831.h5')

probability_model = tf.keras.Sequential([model, tf.keras.layers.ReLU(max_value=1)])

print(labels[0].shape)

gen_image = probability_model.predict(label)

print(gen_image)