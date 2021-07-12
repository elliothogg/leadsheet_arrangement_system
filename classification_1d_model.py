import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with open('training_data.pickle', 'rb') as file:
    training_data = pickle.load(file)

x, y = training_data


# print(x.shape)
# print(y.shape)

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
	tf.keras.layers.Dense(88, activation='relu'),
	tf.keras.layers.Dense(n_outputs)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200)

model.save('saved_model/my_model')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)