import numpy as np
import os
import sys
import inspect
import copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chord_scraper.utils_dict import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation


# This script takes the extracted chord symbols from the data_extractor and outputs a list of chord types for the chord_generator
# it also embeds the chord symbols as 88-note vectors, which were used in a previous iteration of the cDCGAN model

def chords_to_array(leadsheet_data):
    chords = []
    for bar in leadsheet_data['bars']:
        for chord in bar['chords']:
            chords.append(chord)
    return chords

# Gets the chord label from each chord symbol, i.e. "major", "minor-seventh", ..
def extract_chord_labels(chords):
    chord_labels = []
    for chord in chords:
        chord_labels.append(chord['kind'])
    return chord_labels

# Converts each chord symbol to integer notation,  i.e. major would become [0,4,7] (root, major-third, perfect-fifth)
# We ignore the root of the chord, as we are only concerned with the type. We will transpose chords back to their roots when creating arrangement
def convert_chords_to_integer_notation(chords):
    chords_int = []
    for chord in chords:
        chords_int.append({'notes': chord_label_to_integer_notation[chord['kind']].copy(), 'extensions': chord['extensions']}) # we have to make a copy of chord integer array
    return chords_int

# Adds any extensions to the integer notated chords, i.e. a major chord with an added 9th would become [0,2,4,7], with 2 representing the 9th
def add_extensions_to_chord_integer_notation(chords_int_notation):
    chords_int_exten = []
    for chord in chords_int_notation:
        chord_inc_extensions = chord['notes'].copy()
        for extension in chord['extensions']:
            note_int = int(extensions_to_integer_notation[extension['degree']]) + int(extension['alter'])
            type = extension['type']
            if (type == "add"):
                chord_inc_extensions.append(note_int)
            elif (type == "alter"): # if type == alter, then remove note and add altered version
                try:
                    chord_inc_extensions.remove(extensions_to_integer_notation[extension['degree']])
                except Exception:
                    pass
                chord_inc_extensions.append(note_int)
                #convert to dict to remove duplicates then back to list
        chords_int_exten.append(sorted(chord_inc_extensions))
    return chords_int_exten

# Places the integer notated chord in the central octave of an 88-note vector, i.e. starting on middle-C (C4) and ending on B4
def embed_chords_88_vectors(chords):
    embedded_chords = []
    C4 = noteMidiDB['C4'] - 1 # needs to be 0 indexed
    for chord in chords:
        chord_88 = np.zeros(88, "int8")
        for note in chord:
            chord_88[C4 + note] = 1
        embedded_chords.append(chord_88)
    return embedded_chords

# Main method. Takes extracted leadsheet chord symbols and returns list of chord labels
def embed_chords(leadsheet_data):
    chords = chords_to_array(leadsheet_data)
    chord_labels = extract_chord_labels(chords)

    return chord_labels