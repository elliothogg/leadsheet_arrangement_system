import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from note_name_to_number import note_number_to_name

# import test data

# load the training data
def load_real_samples():
    with open('training_data_1d.pickle', 'rb') as file:
        training_data_2d = pickle.load(file)
    real_chords, source = training_data_2d
    print(source.shape)
    print(real_chords.shape)

    return [source, real_chords]



dataset = load_real_samples()
chord_index = 20
source_chord = dataset[0][chord_index]
real_chord = dataset[1][chord_index]


generator = load_model('novel_c_gan.h5')

generated_chord = generator.predict(np.array([source_chord]))


def convert_note_numbers_to_names(note_vector):
    note_names = []
    index = 0
    for note in note_vector:
        if note == 1:
            note_names.append(note_number_to_name[index + 1]) # add 1 to note as dictionary is 1 indexed
        index = index + 1
    return note_names


print("source_chord:", source_chord[39:51])
print("real_chord:", convert_note_numbers_to_names(real_chord))
print("generated_chord:", convert_note_numbers_to_names(generated_chord[0]))

