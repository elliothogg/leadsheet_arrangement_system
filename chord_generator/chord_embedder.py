import pandas as pd
import csv
import os
import copy
import numpy as np
import pickle
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chord_scraper.utils_dict import chord_label_to_integer_notation, extensions_to_integer_notation, noteMidiDB

# This script takes the Jazz-Chords dataset and exports 

def load_pickle(path):
    data = None
    with open(path, 'rb') as file:
        data =  pickle.load(file)
    return data

chords_in_c = load_pickle("../chord_scraper/dataset/chords_in_c.pickle")
out_dir = "./training_data/"
training_data = ()
training_data_1d = ()
training_data_2d = ()
verbose = False


def write_labels_note_numbers():
    with open(out_dir + 'label_note_numbers.csv', mode='w') as csv_file:
        fieldnames = ['label', 'notes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for chord in chords_in_c:
            label = chord['type']
            notes = chord['note_numbers']
            writer.writerow({'label': label, 'notes': notes})


def create_chord_label_vectors():
    index = 0
    invalid_chord_types_index = []
    for chord in chords_in_c:
        label = []
        try:
            label = copy.deepcopy(chord_label_to_integer_notation[chord['type']]) #convert chord label to set of associated notes as integer notations
        except:
            if (verbose): print("Error - unrecognised chord label -", chord['type'], ". Need to add chord label to chord_label_to_integer_notation inside note_name_to_number.py")
            invalid_chord_types_index.append(index)
            index = index + 1
            continue
        if len(chord['extensions']) > 0: #convert any extensions to integer notation and add to label array
            for extension in chord['extensions']:
                try:
                    note_int = int(extensions_to_integer_notation[extension['degree']]) + int(extension['alter'])
                except:
                    if (verbose): print("Error - unrecognised extension integer:", extension['degree'], ". Need to add extension to extensions_to_integer_notation inside note_name_to_number.py")
                    continue
                if (type == "alter"): # if type == alter, then remove note and add altered version
                    try:
                        label.remove(extensions_to_integer_notation[extension['degree']])
                    except Exception:
                        pass
                label.append(note_int)
        del chord['extensions']
        label = list(set(label))
        label.sort()
        chord['label'] = label
        index = index + 1
    invalid_chord_types_index.sort(reverse=True) # reverse the order of the invalid chords indices as need to remove from list in reverse order
    if (verbose): print("total invalid chord types: ", len(invalid_chord_types_index))
    if (verbose): print("deleting now:")
    for index in invalid_chord_types_index:
        if (verbose): print("deleting chord at index", index, ". Type was invalid:")
        del chords_in_c[index]


def convert_notes_to_88_key_vectors():
    global chords_in_c
    for chord in chords_in_c:
        notes = np.zeros(88, dtype='int8')
        for note in chord['note_numbers']:
            notes[note - 1] = 1
        chord['notes'] = notes
        del chord['note_numbers']


# a vector of 12 numbers representing which notes are in the chord
def convert_labels_to_numpy_arrays():
    global chords_in_c
    for chord in chords_in_c:
        label = np.zeros(12, dtype='int8')
        for step in chord['label']:
            label[step] = 1
        chord['label'] = label


def write_label_note_vectors():
    with open(out_dir + 'label_note_vectors.csv', mode='w') as csv_file:
        fieldnames = ['label', 'notes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for chord in chords_in_c:
            label = chord['type']
            notes = "".join(map(str, chord['notes']))
            writer.writerow({'label': label, 'notes': notes})


def main():
    print("Embedding Chord Data...")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    write_labels_note_numbers()
    create_chord_label_vectors()
    convert_notes_to_88_key_vectors()
    convert_labels_to_numpy_arrays() #use binary number instead

    write_label_note_vectors()
    print("Writing Training Data...")



 
    print("Embedding Complete. Find Training Data in " + out_dir)

main()