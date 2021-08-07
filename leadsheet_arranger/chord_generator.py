# from chord_embedder import embed_chords
import pickle
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randint
from numpy.random import randn
import numpy as np
import tensorflow as tf
import os
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from chord_scraper.utils_dict import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation, note_number_to_name, unwanted_chord_tones

#prevents scientific number notation
np.set_printoptions(suppress=True)

def generate_latent_points(n_samples, class_label, latent_dim=100, n_classes=3):
    	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = np.array([class_label] * n_samples)
	return [z_input, labels]


c_gan_model = load_model('../chord_generator/cgan_generator.h5')

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
        chord = chord.flatten()
        chord_num = []
        i = 16 # first note is C2
        for note in chord:
            if note >0.5:
                chord_num.append(i)
            i = i + 1
        chords_as_numbers.append(chord_num)
    return chords_as_numbers


def chord_generator(source_chords):
    generated_chords = generate_chords(source_chords)
    numbered_chords = chord_vectors_to_note_numbers(generated_chords)
    return numbered_chords



def generate_chords():
    dominant_label = 0
    min_7_label = 1
    maj_7_label = 2
    latent_points, labels = generate_latent_points(100, dominant_label)
    dominant_chords = chord_vectors_to_note_numbers(c_gan_model.predict([latent_points, labels]))
    latent_points, labels = generate_latent_points(100, min_7_label)
    min_7_chords = chord_vectors_to_note_numbers(c_gan_model.predict([latent_points, labels]))
    latent_points, labels = generate_latent_points(100, maj_7_label)
    maj_7_chords = chord_vectors_to_note_numbers(c_gan_model.predict([latent_points, labels]))
    return (dominant_chords, min_7_chords, maj_7_chords)

# takes a note in its vector representation (integer between 1-88)
# and converts it to its integer_notation representation (number between 0-11)
def note_as_integer_notation(note):
    return (note + 8) % 12

def calculate_accuracy(chords, invalid_notes):
    total_count = len(chords)
    invalid_count = 0
    for chord in chords:
        for note in chord:
            if note_as_integer_notation(note) in invalid_notes:
                invalid_count += 1
                break
    return 1 - (invalid_count / total_count) 

def calculate_uniqueness_percent(chords):
    total = len(chords)
    num_of_unique = len(set(map(tuple, chords)))
    print(total, num_of_unique)
    return (num_of_unique / total) * 100

def test_generated_chords(chords):
    dominant_chords, min_7_chords, maj_7_chords = chords
    dom_acc = calculate_accuracy(dominant_chords, unwanted_chord_tones['dominant']) 
    min_7_acc = calculate_accuracy(min_7_chords, unwanted_chord_tones['minor-seventh']) 
    maj_7_acc = calculate_accuracy(maj_7_chords, unwanted_chord_tones['major-seventh']) 
    print("--ACCURACY--")
    print("dominant: ", dom_acc)
    print("minor_seventh: ", min_7_acc)
    print("major_seventh: ", maj_7_acc)
    print("--UNIQUE CHORDS--")
    print("dominant: ", calculate_uniqueness_percent(dominant_chords))
    print("minor_seventh: ", calculate_uniqueness_percent(min_7_chords))
    print("major_seventh: ", calculate_uniqueness_percent(maj_7_chords))


def print_chords_note_names(chords):
    note_names = []
    for chord in chords:
        chord_notes = []
        for note in chord:
            chord_notes.append(note_number_to_name[note])
        note_names.append(chord_notes)
    print(note_names)


chords = generate_chords()

test_generated_chords(chords)
print_chords_note_names(chords[2])


