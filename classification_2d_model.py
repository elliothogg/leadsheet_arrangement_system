import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This is the same
with open('training_data_2d.pickle', 'rb') as file:
    training_data_2d = pickle.load(file)

with open('training_data.pickle', 'rb') as file:
    training_data = pickle.load(file)


# the same data represented as: (88,) vs (7,12)[4 notes ommited]  &  (12,) vs (7,12)
x_1d, y_1d = training_data
x_2d, y_2d = training_data_2d


print(x_2d.shape)
print(y_2d.shape)

for i in range(100):
    	# define subplot
	plt.subplot(10, 10, 1 + i)
	# turn off axis
	plt.axis('off')
	# plot raw pixel data
	plt.imshow(x_2d[i], cmap='gray_r')
plt.show()

x_train_2d = x_2d[0:4000]
x_test_2d = x_2d[4000: 4831]
y_train_2d = y_2d[0:4000]
y_test_2d = y_2d[4000: 4831]

x_train_1d = x_1d[0:4000]
x_test_1d = x_1d[4000: 4831]
y_train_1d = y_1d[0:4000]
y_test_1d = y_1d[4000: 4831]

print(x_train_2d.shape)
print(y_train_2d.shape)

# gets the length of x and y vectors for input and output
n_inputs, n_outputs = x_2d.shape[1], y_2d.shape[1]

print(n_inputs, n_outputs)


model = tf.keras.Sequential([
tf.keras.layers.Conv1D(7, 2, activation='relu', input_shape=(7, 12)),
tf.keras.layers.MaxPooling1D(1),
tf.keras.layers.Conv1D(14, 2, activation='relu'),
tf.keras.layers.MaxPooling1D(1),
tf.keras.layers.Conv1D(14, 2, activation='relu'),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(12)
])



model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(x_train_2d, y_train_1d, epochs=200)

model.save('cnn_network')

test_loss, test_acc = model.evaluate(x_test_2d,  y_test_1d, verbose=2)

print('\nTest accuracy:', test_acc)