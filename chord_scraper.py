import re
import os
from note_name_to_number import noteMidiDB, transpose_to_c, extensions_to_integer_notation, chord_label_to_integer_notation
from stored_chords import stored_chords
import copy
import numpy as np
import pickle

chords = stored_chords
chords_in_c = []
training_data = ()
chords_meta_data = {}
index_of_inverted_chords = []
deviation = 20 #allows notes either-side of the chord location to be included in the chord (notes next to each other in a chord are usually ofset on the x-axis)
directory = "./fully_arranged_standards_musicxml" #directory of musicXML files


def alter_note(note, step):
    if step == "1":
        return note + "#"
    if step == "-1":
        return note + "b"
    return note

# sort function for ordering note names
def sort_note_names(note):
    return noteMidiDB[note]

def sort_notes():
    global chords
    for chord in chords:
        chord['note_numbers'].sort()
        chord['notes'].sort(key=sort_note_names)

def note_deviation_limits(deviation, location):
    location_int = float(location)
    lower_limit = location_int - deviation
    upper_limit = location_int + deviation
    return [lower_limit, upper_limit]

def note_within_deviation_limits(deviation_limits, location):
    location_int = float(location)
    if (location_int > deviation_limits[0]) and (location_int < deviation_limits[1]):
        return True
    return False

def return_invalid_chords(chords):
    invalid_chord_indices = []
    i = 0
    for chord in chords:
        if (len(chord['notes']) < 3) or (len(set(chord['notes'])) <3):
            invalid_chord_indices.append(i)
        i = i + 1
    return invalid_chord_indices


def get_chords_meta(file_name):
    harmony_found = False
    line_number = 0
    chord_root = ""
    chord_type = ""
    degree_alter = ""
    extensions = []
    extension = ""
    root_alter= ""
    current_songs_chords = []
    with open(file_name) as topo_file:
        for line in topo_file:
            line_number += 1
            if "<measure number" in line: # try an xml parser https://docs.python.org/3/library/xml.etree.elementtree.html
                measure = re.search(r"(?<=number=\")(.*?)(?=\")", line).group(1)
            if "<fifths" in line:
                fifths = re.search(r"(?<=>)(.*?)(?=<\/fifths>)", line).group(1)
            if "<root-step" in line:
                extensions = []
                harmony_found = True
                chord_root = re.search(r"(?<=>)(.*?)(?=<\/root-step>)", line).group(1)
            if "<root-alter" in line:
                root_alter = re.search(r"(?<=>)(.*?)(?=<\/root-alter>)", line).group(1)
                chord_root = alter_note(chord_root, root_alter)
            if ("<kind" in line) and (harmony_found):
                chord_type = re.search(r"(?<=>)(.*?)(?=<\/kind>)", line).group(1)
                if (chord_type =="other") and (re.search(r"(?<=text=\")(.*?)(?=\")", line).group(1) == "°"):
                    chord_type = "diminished"
            if ("<degree-value" in line) and (harmony_found):
                extension = re.search(r"(?<=>)(.*?)(?=<\/degree-value>)", line).group(1)
            if ("<degree-alter" in line) and (harmony_found):
                degree_alter = re.search(r"(?<=<degree-alter>)(.*?)(?=<\/degree-alter>)", line).group(1)
                extension = alter_note(extension, degree_alter)
            if ("<degree-type" in line) and (harmony_found):
                degree_type = re.search(r"(?<=>)(.*?)(?=<\/degree-type>)", line).group(1)
                if degree_type != "subtract":
                    extensions.append(extension)
            if ("<note default-x" in line) and (harmony_found ):
                chord_x_location = re.search(r"(?<=default-x=\")(.*?)(?=\")", line).group(1)
                chord_y_location = re.search(r"(?<=default-y=\")(.*?)(?=\")", line).group(1)
                current_songs_chords.append({'root_note': chord_root, 'type': chord_type, 'extensions': extensions, 'notes_x_location': chord_x_location, 'notes_y_location': chord_y_location, 'measure': measure, 'notes': [], 'song': file_name, 'fifths': fifths})
                harmony_found = False
    get_chord_notes(file_name, current_songs_chords)

def get_chord_notes(file_name, current_songs_chords):
    note_found = False
    for chord in current_songs_chords:
        with open(file_name) as topo_file:
            for line in topo_file:
                if "<measure number" in line:
                    measure = re.search(r"(?<=number=\")(.*?)(?=\")", line).group(1)
                if ("<note default-x" in line) and (measure == chord['measure']):
                    note_x = re.search(r"(?<=default-x=\")(.*?)(?=\")", line).group(1)
                    if note_within_deviation_limits(note_deviation_limits(deviation, chord['notes_x_location']), note_x):
                        note_found = True
                if ("<step>" in line) and (note_found):
                    note = re.search(r"(?<=<step>)(.*?)(?=<\/step>)", line).group(1)
                if ("<alter>" in line) and (note_found):
                    alter = re.search(r"(?<=<alter>)(.*?)(?=<\/alter>)", line).group(1)
                    note = alter_note(note, alter)
                if ("<octave>" in line) and (note_found):
                    octave = re.search(r"(?<=<octave>)(.*?)(?=<\/octave>)", line).group(1)
                    note = note + octave
                    chord['notes'].append(note)
                    note_found = False
    chords.append(current_songs_chords)
                    
def remove_invalid_chords():
    global chords 
    invalid_chords = return_invalid_chords(chords)
    invalid_chords.sort(reverse=True) # reverse the order of the invalid chords indices as need to remove from list in reverse order
    for index in invalid_chords:
        del chords[index]

def count_degree_tags(file_name): #checks there is an equal amount of degree-value/alter/type tags
    with open(file_name) as topo_file:
        num_degree_val = 0
        num_degree_alt = 0
        num_degree_type = 0
        for line in topo_file:
            if ("<degree-value" in line):
                num_degree_val = num_degree_val + 1
            if ("<degree-alter" in line):
                num_degree_alt = num_degree_alt + 1
            if ("<degree-type" in line):
                num_degree_type = num_degree_type + 1
    print(file_name + " " + "degree tags equal? ", num_degree_val, num_degree_alt, num_degree_type, (num_degree_val == num_degree_alt == num_degree_type))


def print_chords_with_extensions(chords):
    print("\nCHORDS WITH EXTENSIONS:\n")
    for chord in chords:
        if len(chord['extensions']):
            print(chord)


def add_note_numbers():
    global chords 
    for chord in chords:
        note_numbers = []
        for note in chord['notes']:
            note_numbers.append(noteMidiDB[note])
        chord['note_numbers'] = note_numbers
  

def mine_chords_from_dir(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".musicxml"):
            print(os.path.join(directory, filename))
            # count_degree_tags(filename) # checks if degree tags are even, essential for extensions to be gathered correctly
            get_chords_meta(os.path.join(directory, filename))

def flatten_chords():
    global chords 
    chords = [item for sublist in chords for item in sublist]


def gather_chord_type_meta_data():
    for chord in stored_chords:
        if chord['type'] in chords_meta_data:
            chords_meta_data[chord['type']] = chords_meta_data[chord['type']] + 1
        else: chords_meta_data[chord['type']] = 1
    print(chords_meta_data)

def print_num_chords():
    print(len(chords))

def chords_pretty_print(chords_in):
    j=0
    for chord in chords_in:
        print(j, chord)
        j = j + 1

def count_num_inverted_chords():
    global chords
    global index_of_inverted_chords
    index_of_inverted_chords = []
    count = 0
    index = 0
    for chord in chords:
        root_note = chord['root_note']
        bottom_note = chord['notes'][0]
        if len(bottom_note) == 3:
            bottom_note = chord['notes'][0][0:2]
        else: bottom_note = chord['notes'][0][0]
        if transpose_to_c[root_note] != transpose_to_c[bottom_note]: #some root notes are "F#" with a bottom note of "Gb". This method ensures that these are evaluated as equal
            count = count + 1
            index_of_inverted_chords.append(index)
        index = index + 1
    print ("number of inverted chords: ", count)
    return count

def transpose_chords_to_key_c():
    global chords
    global chords_in_c
    c2_num = noteMidiDB['C2']
    c3_num = noteMidiDB['C3']
    chords_in_c = copy.deepcopy(chords)
    for chord in chords_in_c:
        del chord['notes_x_location']
        del chord['notes_y_location']
        del chord['measure']
        del chord['song']
        del chord['fifths']
        del chord['notes'] #remove all keys apart from type, extensions, and note_numbers
        lowest_note = chord['note_numbers'][0]
        root = chord['root_note']
        if lowest_note >= c3_num:
            gap = transpose_to_c[root]["down"]
        elif lowest_note > 20 or lowest_note < c2_num: # if root note in chord is F2 or above, transpose the chord so the root is C3
            gap = transpose_to_c[root]["up"]
        else:
            gap = transpose_to_c[root]["down"]
        for idx, note in enumerate(chord['note_numbers']):
            chord['note_numbers'][idx] = chord['note_numbers'][idx] + gap

        del chord['root_note']
            

# the number of chords with C NOT as bottom note should be = total num inverted chords
def test_transpose_to_c():
    index_of_inverted_transposed_chords = []
    num_of_inverted_chords = count_num_inverted_chords()
    count = 0
    bottom_note_counts = {}
    index = 0
    for chord in chords_in_c:
        if chord['note_numbers'][0] not in [4, 16, 28, 40, 52, 64, 76]:
            count = count + 1
            index_of_inverted_transposed_chords.append(index)
        if chord['note_numbers'][0] in bottom_note_counts:
            bottom_note_counts[chord['note_numbers'][0]] = bottom_note_counts[chord['note_numbers'][0]] + 1
        else:
            bottom_note_counts[chord['note_numbers'][0]] = 1
        index = index + 1
    if count != num_of_inverted_chords:
        print ("error - transpose function not working: expected num of inverted chords: ", num_of_inverted_chords, ". actual ", count)
        print ("chords with errors:")
        find_incorrectly_transposed_chords(index_of_inverted_transposed_chords)
        return False
    else:
        print ("success - transpose working correctly")
        return True

def find_incorrectly_transposed_chords(index_of_inverted_transposed_chords):
    for chord in index_of_inverted_chords:
        if chord not in index_of_inverted_transposed_chords:
            print(chords[chord])


def transpose_extreme_octaves():
    global chords_in_c
    for chord in chords_in_c:
        if chord['note_numbers'][0] < 16:
            for idx, note in enumerate(chord['note_numbers']):
                chord['note_numbers'][idx] = chord['note_numbers'][idx] + 12
        if chord['note_numbers'][0] > 40:
            for idx, note in enumerate(chord['note_numbers']):
                chord['note_numbers'][idx] = chord['note_numbers'][idx] - 12
    while test_transpose_extreme_octaves(False) == False: # recursively runs function until all chord roots are within desirable range (16-40)
        transpose_extreme_octaves()
    index = 0
    out_of_range_chords = []
    for chord in chords_in_c:
        note = chord['note_numbers']
        if note[len(note)-1] > 88:
            out_of_range_chords.append(index)
        index = index + 1
    print(len(out_of_range_chords), "chords have outlying voicings. Removing from array:")
    out_of_range_chords.sort(reverse=True) # reverse the order of the invalid chords indices as need to remove from list in reverse order
    for index in out_of_range_chords:
        del chords_in_c[index]
        print("chord at index ", index, "deleted")
    print(len(chords_in_c))
    


def test_transpose_extreme_octaves(is_user_test=True):
    global chords_in_c
    for chord in chords_in_c:
        if chord['note_numbers'][0] < 16 or chord['note_numbers'][0] > 40:
            if is_user_test: print("Error - some chords still in extreme octaves")
            return False
    if is_user_test: print("Success - all chords in desirable range")
    return True
    
def create_chord_label_vectors():
    index = 0
    invalid_chord_types_index = []
    for chord in chords_in_c:
        label = []
        if len(chord['extensions']) > 0: #convert any extensions to integer notation and add to label array
            for extension in chord['extensions']:
                error = False
                modifier = 0
                if "b" in extension[-1] or "#" in extension[-1]:
                    accidental = extension[-1]
                else: accidental = False
                string_ofset = 1 if accidental else 0
                scale_degree = extension[0: (len(extension) - string_ofset)]
                integer_notation  = 0
                try:
                    integer_notation = extensions_to_integer_notation[scale_degree]
                except:
                    print("Error - unrecognised extension integer:", scale_degree, ". Need to add extension to extensions_to_integer_notation inside note_name_to_number.py")
                    error = True
                if (accidental):
                    modifier = 1 if accidental == "#" else -1
                if error == False:
                    label.append(integer_notation + modifier)
        del chord['extensions']
        try:
            label = label + chord_label_to_integer_notation[chord['type']] #convert chord label to set of associated notes as integer notations
        except:
            print("Error - unrecognised chord label -", chord['type'], ". Need to add chord label to chord_label_to_integer_notation inside note_name_to_number.py")
            invalid_chord_types_index.append(index)
        label.sort()
        chord['label'] = label
        del chord ['type']
        index = index + 1
    invalid_chord_types_index.sort(reverse=True) # reverse the order of the invalid chords indices as need to remove from list in reverse order
    print("total invalid chord types: ", len(invalid_chord_types_index))
    print("deleting now:")
    for index in invalid_chord_types_index:
        print("deleting chord at index", index, ". Type was invalid:")
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


def write_training_data():
    training_data_pickle = pickle.dumps(training_data)
    with open('training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)

def main():
    # mine_chords_from_dir(directory)
    # flatten_chords()
    # remove_invalid_chords()
    # add_note_numbers()
    # sort_notes()

    chords_pretty_print(chords)

    count_num_inverted_chords()
    transpose_chords_to_key_c()
    test_transpose_to_c()
    transpose_extreme_octaves()
    test_transpose_extreme_octaves()

    
    create_chord_label_vectors()

    convert_notes_to_88_key_vectors()
    convert_labels_to_numpy_arrays() #use binary number instead


    create_training_data()
    chords_pretty_print(chords_in_c)

    write_training_data()

main()

gather_chord_type_meta_data()


