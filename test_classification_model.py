import tensorflow as tf
import pickle
import numpy as np
with open('training_data.pickle', 'rb') as file:
    training_data = pickle.load(file)

x, y = training_data

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()

probability_model = tf.keras.Sequential([new_model, tf.keras.layers.ReLU()])

print(x[0])

test = x[0]

prediction = probability_model.predict(x)

print(prediction[4720])

print(y[4720])
