import tensorflow as tf
import pickle
import copy
import numpy as np
from note_name_to_number import note_number_to_name

with open('training_data.pickle', 'rb') as file:
    training_data_1d = pickle.load(file)

with open('training_data_2d.pickle', 'rb') as file:
    training_data_2d = pickle.load(file)



x_1d, y_1d = training_data_1d
x_2d, y_2d = training_data_2d

cnn_model = tf.keras.models.load_model('cnn_network')
cnn_model.summary()

# add a Rectified Linear Unit (ReLU) activation function - makes all negative values 0 and positive the same
probability_model = tf.keras.Sequential([cnn_model, tf.keras.layers.ReLU(max_value=1)])

#converts 7x12 matrix back to 88 note vector (adds A0, Bb0, B0, & C8 back)
def convert_note_matrix_to_note_vector(note_matrix):
    note_vector = [item for sublist in note_matrix for item in sublist]
    note_vector.insert(0, 0)
    note_vector.insert(0, 0)
    note_vector.insert(0, 0)
    note_vector.append(0)
    return note_vector

def convert_note_numbers_to_names(note_vector):
    note_names = []
    index = 0
    for note in note_vector:
        if note == 1:
            note_names.append(note_number_to_name[index + 1]) # add 1 to note as dictionary is 1 indexed
        index = index + 1
    return note_names


prediction = probability_model.predict(x_2d)

index = 4718
chord = convert_note_numbers_to_names(convert_note_matrix_to_note_vector(x_2d[index]))
actual_label = y_1d[index]
predicted_label = np.rint(prediction[index])

print("chord: ", chord)
print("actual_label: ", actual_label)
print("predic_label: ", predicted_label)
