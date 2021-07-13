import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# There are 2 different 1D representations of the training/test data
# 1) training_data = labels (12,) | notes (88,)   label vectors represent one octave of a piano, with all chord tones included in it
#												  note vectors represent the whole piano

# 2) training_data_1d = labels (88,) | notes (88,) 12 note label vectors are embedded into 88 note array, sitting in C4 (middle C) octave


# The purpose of this RNN classification network is to see the effect of the differing representations of the labels. Conditional Generative Adversarial
# Networks require the labeled and projected data to be of the same shape.


data_pickle_path = 'training_data.pickle'

with open(data_pickle_path, 'rb') as file:
    training_data = pickle.load(file)

x, y = training_data



x_train = x[0:4000]
x_test = x[4000: 4831]

y_train = y[0:4000]
y_test = y[4000: 4831]

print(x_train.shape)
print(y_train.shape)

# gets the length of x and y vectors for input and output
n_inputs, n_outputs = x.shape[1], y.shape[1]

print(n_inputs, n_outputs)


model = tf.keras.Sequential([
	tf.keras.layers.Dense(n_inputs, activation='relu'),
	tf.keras.layers.Dense(176, activation='relu'),
	tf.keras.layers.Dense(176, activation='relu'),
	tf.keras.layers.Dense(88, activation='relu'),
	tf.keras.layers.Dense(n_outputs)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500)

model.save('rnn_network')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)