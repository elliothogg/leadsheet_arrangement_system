import numpy as np
import os
import sys
import inspect
import copy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from note_name_to_number import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation


def chords_to_array(leadsheet_data):
    chords = []
    for bar in leadsheet_data['bars']:
        for chord in bar['chords']:
            chords.append(chord)
    return chords

# We ignore the root of the chord, as we are only concerned with the type. We will transpose chords back to their roots when creating arrangement
def convert_chords_to_integer_notation(chords):
    chords_int = []
    for chord in chords:
        chords_int.append({'notes': chord_label_to_integer_notation[chord['kind']].copy(), 'extensions': chord['extensions']}) # we have to make a copy of chord integer array
    return chords_int

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

def embed_chords_88_vectors(chords):
    embedded_chords = []
    C4 = noteMidiDB['C4'] - 1 # needs to be 0 indexed
    for chord in chords:
        chord_88 = np.zeros(88, "int8")
        for note in chord:
            chord_88[C4 + note] = 1
        embedded_chords.append(chord_88)
    return embedded_chords


def embed_chords(leadsheet_data):
    chords = chords_to_array(leadsheet_data)
    chords_int = convert_chords_to_integer_notation(chords)

    chords_int_ext = add_extensions_to_chord_integer_notation(chords_int)

    embedded_chords = embed_chords_88_vectors(chords_int_ext)

    return embedded_chords