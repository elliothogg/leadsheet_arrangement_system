import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_pickle_path = '../../chord_vectors_training_data.pickle'

with open(data_pickle_path, 'rb') as file:
    training_data = pickle.load(file)

y, x = training_data
n_inputs = x.shape[1]

model = tf.keras.Sequential([
	tf.keras.layers.Dense(n_inputs, activation='relu'),
	tf.keras.layers.Dense(88, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(176, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(176, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(88, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(x, y, validation_split=0.33, epochs=300, batch_size=1028)
# tf.keras.utils.plot_model(model, "deep_feedforward_network.png", show_shapes=True, show_layer_names=True)
# model.summary()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
