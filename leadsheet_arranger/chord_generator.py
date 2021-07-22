from chord_embedder import embed_chords
import pickle
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np
import tensorflow as tf
import os
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from note_name_to_number import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation

#prevents scientific number notation
np.set_printoptions(suppress=True)

c_gan_model = load_model('novel_c_gan.h5')

# takes a list of embedded chord labels (source chords) and returns a list of generated chords

# **need to round note predictions to 0 or 1
def generate_chords(source_chords, c_gan_model=c_gan_model):
    generated_chords = []
    for chord in source_chords:
        generated_chords.append(c_gan_model.predict(np.array([chord])))
    return generated_chords




def chord_vectors_to_note_numbers(chords):
    chords_as_numbers = []
    for chord in chords:
        chord = chord[0]
        chord_num = []
        i = 1 # note number dicts are 1 indexed
        for note in chord:
            if note == 1:
                chord_num.append(i)
            i = i + 1
        chords_as_numbers.append(chord_num)
    return chords_as_numbers


def chord_generator(source_chords):
    generated_chords = generate_chords(source_chords)
    numbered_chords = chord_vectors_to_note_numbers(generated_chords)
    return numbered_chords