import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_pickle_path = 'chord_matrices_training_data.pickle'

with open(data_pickle_path, 'rb') as file:
    training_data = pickle.load(file)

y, x = training_data

def plot_chord_images():
    for i in range(20):
            # define subplot
        plt.subplot(5, 5, 1 + i)
        plt.tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False,labelleft=False)
        # plot raw pixel data
        plt.imshow(x[i + 200], cmap='GnBu')
    plt.show()


# common pattern: a stack of Conv1d and maxpooling1d layers
model_2 = tf.keras.Sequential([
tf.keras.layers.Conv1D(7, 2, activation='relu', input_shape=(7, 12)),
tf.keras.layers.MaxPooling1D(1),
tf.keras.layers.Conv1D(14, 2, activation='relu'),
tf.keras.layers.MaxPooling1D(1),
tf.keras.layers.Conv1D(14, 2, activation='relu'),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(3)
])

# here we use 64 parallel feature maps and kernal size of 3.
model = tf.keras.Sequential([
tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(7, 12)),
tf.keras.layers.MaxPooling1D(pool_size=1),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
tf.keras.layers.MaxPooling1D(pool_size=1),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
tf.keras.layers.MaxPooling1D(pool_size=1),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(84, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(3)
])


# we have to use BinaryCrossentropy as values are binary (get more info on this) cross entropy loss function as multi-class classification problem
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x, y, validation_split=0.33, epochs=300, batch_size=1028)

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