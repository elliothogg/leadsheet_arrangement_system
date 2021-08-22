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
tf.get_logger().setLevel('INFO')
import os
import inspect
import sys
import random
import copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chord_scraper.utils_dict import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation, note_number_to_name, unwanted_chord_tones



# This is the chord_generator subsystem. It takes chord labels, i.e. "major-seventh", embeds them as integer labels, and passes them to the trained cDCGAN model
# It then converts the cDCGAN's predictions into note number lists and returns them to the leadsheet_arranger
# For further information see report - section 6.1.2


#contains all valid chord types + does some converting to similar types to enable max capacity
chord_types_dict = {
    "dominant": "dominant",
    "dominant-ninth": "dominant",
    "dominant-11th": "dominant",
    "dominant-13th": "dominant",
    "minor": "minor-seventh",
    "minor-sixth": "minor-seventh",
    "minor-seventh": "minor-seventh",
    "major": "major-seventh",
    "major-sixth": "major-seventh",
    "major-seventh": "major-seventh",
    "major-ninth": "major-seventh"
}

# Load trained cDCGAN model (relative path is different depending on where functions are called)

try:
    c_gan_model = load_model('../chord_generator/cDCGAN/cgan_generator.h5')
except:
    pass
try:
    c_gan_model = load_model('cgan_generator.h5')
except:
    pass

#prevents scientific number notation
np.set_printoptions(suppress=True)

# Generates random noise vectors that are passed as input to the trained cDCGAN model in order to generate varied chords
def generate_latent_points(n_samples, class_label, latent_dim=100, n_classes=3):
    	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = np.array([class_label] * n_samples)
	return [z_input, labels]




# takes a list of embedded chord labels (source chords) and returns a list of generated chords
def generate_chords(source_chords, c_gan_model=c_gan_model):
    generated_chords = []
    for chord in source_chords:
        generated_chords.append(c_gan_model.predict(np.array([chord])))
    return generated_chords

# Converts cDCGAN predictions (3x12 vectors) into note number lists
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

# Assigns generated chords to input chord type labels
def assign_chords(leadsheet_chord_labels, chords, verbose):
    dominant_chords, min_7_chords, maj_7_chords = chords
    leadsheet_chords = []
    for chord_label in leadsheet_chord_labels:
        valid_label = validate_chord_label(chord_label)
        if(valid_label == False):
            leadsheet_chords.append([28,40])
            if (verbose): print("error - ", chord_label, "not yet supported :(. Please manually arrange this chord")
        elif (valid_label == "dominant"):
            leadsheet_chords.append(random.choice(dominant_chords).copy())
        elif (valid_label == "minor-seventh"):
            leadsheet_chords.append(random.choice(min_7_chords).copy())
        elif (valid_label == "major-seventh"):
            leadsheet_chords.append(random.choice(maj_7_chords).copy())
    return leadsheet_chords


# Checks each chord label to see if it is supported by generator
def validate_chord_label(chord_label):
    try:
        return chord_types_dict[chord_label]
    except:
        return False

# Generates bank of chords using trained model. These chords are then assigned to inputs
def generate_chords(model=c_gan_model):
    dominant_label = 0
    min_7_label = 1
    maj_7_label = 2
    latent_points, labels = generate_latent_points(50, dominant_label)
    dominant_chords = chord_vectors_to_note_numbers(model.predict([latent_points, labels]))
    latent_points, labels = generate_latent_points(50, min_7_label)
    min_7_chords = chord_vectors_to_note_numbers(model.predict([latent_points, labels]))
    latent_points, labels = generate_latent_points(50, maj_7_label)
    maj_7_chords = chord_vectors_to_note_numbers(model.predict([latent_points, labels]))
    return (dominant_chords, min_7_chords, maj_7_chords)

# Takes a note in its vector representation (integer between 1-88)
# and converts it to its integer_notation representation (number between 0-11)
def note_as_integer_notation(note):
    return (note + 8) % 12

# Perceivable metrics functions used by cDCGAN model after each training iteration
# Takes a list of generated chords and measures how accurate they are 
# See report section 5.4.3 for more information
def calculate_accuracy(chords, invalid_notes):
    total_count = len(chords)
    invalid_count = 0
    for chord in chords:
        for note in chord:
            if note_as_integer_notation(note) in invalid_notes:
                invalid_count += 1
                break
    return 1 - (invalid_count / total_count) 

# Perceivable metrics functions used by cDCGAN model after each training iteration
# Takes a list of generated chords and measures how unique they are 
# See report section 5.4.3 for more information
def calculate_uniqueness(chords):
    total = len(chords)
    num_of_unique = len(set(map(tuple, chords)))
    return (num_of_unique / total)

# Calls perceivable metrics functions and prints the results
# This function is called by the cDCGAN after each training iteration
# See report section 5.4.3 for more information
def test_generated_chords(chords):
    dominant_chords, min_7_chords, maj_7_chords = chords
    dom_acc = calculate_accuracy(dominant_chords, unwanted_chord_tones['dominant']) 
    min_7_acc = calculate_accuracy(min_7_chords, unwanted_chord_tones['minor-seventh']) 
    maj_7_acc = calculate_accuracy(maj_7_chords, unwanted_chord_tones['major-seventh']) 
    dom_uniq = calculate_uniqueness(dominant_chords)
    min_7_uniq = calculate_uniqueness(min_7_chords)
    maj_7_uniq = calculate_uniqueness(maj_7_chords)
    print('accuracy: dominant=%.2f, minor_seventh=%.2f, major_seventh=%.2f' % (dom_acc, min_7_acc, maj_7_acc))
    print('uniqueness: dominant=%.2f, minor_seventh=%.2f, major_seventh=%.2f' % (dom_uniq, min_7_uniq, maj_7_uniq))
    return dom_acc, min_7_acc, maj_7_acc, dom_uniq, min_7_uniq, maj_7_uniq

# Utility function for error checking
def print_chords_note_names(chords):
    note_names = []
    for chord in chords:
        chord_notes = []
        for note in chord:
            chord_notes.append(note_number_to_name[note])
        note_names.append(chord_notes)
    print(note_names)


# Main method. Called within leadsheet_arranger subsystem. chord labels are passed in and generated chords are returned
def chord_generator(leadsheet_chord_labels, verbose):
    #generate 50 of each chord to be selected
    chords = generate_chords()
    leadsheet_chords = assign_chords(leadsheet_chord_labels, chords, verbose)
    return leadsheet_chords