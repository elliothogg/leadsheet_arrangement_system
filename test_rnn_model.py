import tensorflow as tf
import pickle
import numpy as np
with open('training_data.pickle', 'rb') as file:
    training_data = pickle.load(file)

x, y = training_data

rnn_model = tf.keras.models.load_model('rnn_network')
rnn_model.summary()

probability_model = tf.keras.Sequential([rnn_model, tf.keras.layers.ReLU(max_value=1)])

# utility function that converts 88 note vector to note numbers. Makes it easier for analysis of chords
def convert_note_vector_to_note_numbers(note_vector):
    i = 0
    note_numbers = []
    for note in note_vector:
        if note == 1:
            note_numbers.append(i + 1)
        i = i + 1
    return note_numbers


prediction = probability_model.predict(x)

print(convert_note_vector_to_note_numbers(x[4722]))

print(prediction[4722])

print(y[4722])
