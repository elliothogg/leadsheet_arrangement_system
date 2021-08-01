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

def load_chords():
    chords = []
    with open('../chord_scraper/datasettest/chords_in_c.csv') as f:
        chords = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    return chords

chords_in_c = pd.read_csv('../chord_scraper/datasettest/chords_in_c.csv')
out_dir = "./training_data"
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


# this creates two arrays: trainX - (num of items, 88) . Train y - (num of items, 12)
def create_training_data():
    global training_data
    number_of_items = len(chords_in_c)
    trainX = np.empty([number_of_items, 88]) #notes
    trainy = np.empty([number_of_items, 12]) #labels

    index = 0

    for chord in chords_in_c:
        trainX[index] = chord['notes']
        trainy[index] = chord['label']
        index = index + 1
    training_data = (trainX, trainy)


# this creates a 1D representation of the labels and chord notes in the shape of (88,). The 12 note label vectors are embedded into an 88 note vector, sitting in the middle C octave
def create_1d_training_data():
    global training_data_1d
    number_of_items = len(chords_in_c)
    trainX = np.empty([number_of_items, 88]) #notes
    trainy = np.empty([number_of_items, 88]) #labels

    index = 0

    for chord in chords_in_c:
        trainX[index] = chord['notes'] # assign 88 chord notes array to trainX data

        label_88 = np.zeros(76, dtype='int8') # length 76 to account for 12 note label vector to be added
        C4 = noteMidiDB['C4'] - 1 # needs to be 0 indexed
        label_88 = np.insert(label_88, C4, chord['label']) # inserts the 12 note label vector at the position of C4
        trainy[index] = label_88
        index = index + 1
    training_data_1d = (trainX, trainy)


# this creates a 2d representation of the training data: trainX_2d - (num of items, 7, 12) the bottom 3 notes and top note of the piano are disregardedm giving 7 full octaves
#                                                        trainy_2d - (num of items, 7, 12) the original 12 note vector is transformed into the same shape as trainy, with that vector sitting in the middle C octave
# cGan training generally requires the source and expected data to be of the same shape
def create_2d_training_data():
    global training_data_2d
    number_of_items = len(chords_in_c)
    trainX_2d = np.empty([number_of_items, 7, 12]) #notes
    trainy_2d = np.zeros([number_of_items, 7, 12]) #labels
    index = 0
    for chord in chords_in_c:

        # convert 88 note vector into 7 x 12 note vector
        notes = chord['notes'][3:87] # ommiting bottom 3 notes and top note, leaving 7 full octaves
        i = 0
        for j in range(0, notes.shape[0], 12): # loop through each set of 12 notes
            trainX_2d[index][i] = notes[j: j+12] # assign each set of 12 notes to an array in trainX_2d
            i = i + 1
        
        # convert 1d 12 label vector into 7 x 12 vector
        label = chord['label']
        trainy_2d[index][3] = label

        index = index + 1
    training_data_2d = (trainX_2d, trainy_2d)


def write_training_data():
    training_data_pickle = pickle.dumps(training_data)
    training_data_2d_pickle = pickle.dumps(training_data_2d)
    with open('training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)
    with open('training_data_1d.pickle', 'wb') as file:
        pickle.dump(training_data_1d, file)
    with open('training_data_2d.pickle', 'wb') as file:
        pickle.dump(training_data_2d, file)


def main():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    write_labels_note_numbers()
    create_chord_label_vectors()
    
    
    convert_notes_to_88_key_vectors()
    convert_labels_to_numpy_arrays() #use binary number instead
    
    write_label_note_vectors()

    create_training_data()
    create_1d_training_data()
    create_2d_training_data()
    write_training_data()

print(list(chords_in_c))