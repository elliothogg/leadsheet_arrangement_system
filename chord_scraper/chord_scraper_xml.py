import re
import os
from utils_dict import noteMidiDB, transpose_to_c, extensions_to_integer_notation, chord_label_to_integer_notation, alter_dict
from stored_chords import stored_chords
import copy
import numpy as np
import pickle
import csv
import argparse
import xml.etree.ElementTree as ET


chords = []
chords_in_c = []
training_data = ()
training_data_1d = ()
training_data_2d = ()
chords_meta_data = {}
index_of_inverted_chords = []
deviation = 20 #allows notes either-side of the chord location to be included in the chord (notes next to each other in a chord are usually ofset on the x-axis)
directory = "./fully_arranged_standards_musicxml" #directory of musicXML files
out_dir = ""
verbose = False
meta = False

def set_deviation(deviation_amm):
    global deviation
    deviation = deviation_amm
    
def set_in_directory(direc):
    global directory
    directory = direc
    
def set_out_directory(direc):
    global out_dir
    out_dir = direc

def set_verbose(bool):
    global verbose
    verbose = bool

def set_meta(bool):
    global meta
    meta = bool

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
    current_songs_chords = []
    tree = ET.parse(file_name)
    root = tree.getroot()
    all_bars = root.find('part').findall('measure')
    for bar in all_bars:
        bar_chords = get_chords_from_bar(bar)
        if (bar_chords == None): continue
        for chord in bar_chords:
            current_songs_chords.append(chord)
    get_chord_notes(file_name, current_songs_chords)
    

def get_chords_from_bar(bar):
    bar_number = bar.attrib['number']
    bar_chords = []
    chords = bar.findall('harmony')
    if (len(list(chords)) == 0): return
    for chord_ele in chords:
        extensions = {}
        harmony_ele = chord_ele
        degree_eles = harmony_ele.findall('degree') # each <degree> element contains a chord extension
        chord = convert_root_ele_to_chord_sym(harmony_ele)
        if (chord == None):
            continue
        chord["extensions"] = []
        notes_location = get_chord_notes_location(harmony_ele, bar)
        if (notes_location == None): continue
        chord["notes_x_location"] = notes_location
        chord["measure"] = bar_number
        chord["notes"] = []

        if (degree_eles):
            extensions = extract_extensions(degree_eles)
            chord["extensions"] = extensions
        
        bar_chords.append(chord)
    return bar_chords

def get_chord_notes_location(harmony_ele, bar):
    harmony_ele_index = list(bar).index(harmony_ele)
    number_of_eles_in_bar = (len(list(bar)))

    for i in range(harmony_ele_index, number_of_eles_in_bar, 1):
        if (bar[i].tag == "note" and len(list(bar[i].attrib)) > 1): # if element is note and has x and y note attributes then return default-x
            return bar[i].attrib["default-x"]
    # if note element is not found, then print error
    if (verbose): print('Error - x-location of chord symbol could not be located. Chord will be removed')
    return None


def extract_extensions(degree_eles):
    extensions = []
    for degree in degree_eles:
        value = degree.find('degree-value').text
        if (value == None): continue # if degree-value is empty, do not add extension to extensions list
        alter = ""
        try:
            alter = degree.find('degree-alter').text
        except Exception:
            pass
        degree_type = degree.find('degree-type').text
        extensions.append({"degree": value, "alter": alter, "type": degree_type})
    return extensions

def convert_root_ele_to_chord_sym(harmony_ele):
    root_ele = harmony_ele.find('root') # <root> element encapsulates chord symbols
    try:
        chord_root = root_ele.find('root-step').text
    except:
        return None
    root_adjust = ""
    try:
        root_adjust = root_ele.find('root-alter').text
    except Exception:
        pass
    if (root_adjust):
        chord_root = chord_root + alter_dict[root_adjust]
    chord_type = harmony_ele.find('kind').text
    bass_note = ""
    bass_adjust = ""
    try:
        bass_note = harmony_ele.find('bass').find('bass-step').text
    except Exception:
        pass
    try:
        bass_adjust = harmony_ele.find('bass').find('bass-alter').text
    except Exception:
        pass
    if (bass_adjust):
        bass_note = bass_note + alter_dict[bass_adjust]
    return {'root_note': chord_root, 'type': chord_type}


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
    if (verbose):
        print(file_name + " " + "degree tags equal? ", num_degree_val, num_degree_alt, num_degree_type, (num_degree_val == num_degree_alt == num_degree_type))

# utility function
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
    print()
    print("Chord scraper has finished. Please find output at" + out_dir)

def flatten_chords():
    global chords 
    chords = [item for sublist in chords for item in sublist]


def gather_chord_type_meta_data():
    for chord in chords:
        if chord['type'] in chords_meta_data:
            chords_meta_data[chord['type']] = chords_meta_data[chord['type']] + 1
        else: chords_meta_data[chord['type']] = 1
    print(chords_meta_data)

def print_num_chords():
    print(len(chords))


# utility function
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
    if (verbose): print ("Number of inverted chords: ", count)
    return count

def transpose_chords_to_key_c():
    global chords
    global chords_in_c
    c2_num = noteMidiDB['C2']
    c3_num = noteMidiDB['C3']
    chords_in_c = copy.deepcopy(chords)
    for chord in chords_in_c:
        del chord['notes_x_location']
        del chord['measure']
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
        if (verbose): print ("Error - transpose function not working: expected num of inverted chords: ", num_of_inverted_chords, ". actual ", count)
        if (verbose): print ("Chords with errors:")
        find_incorrectly_transposed_chords(index_of_inverted_transposed_chords)
        return False
    else:
        print ("success - transpose working correctly")
        return True

def find_incorrectly_transposed_chords(index_of_inverted_transposed_chords):
    for chord in index_of_inverted_chords:
        if chord not in index_of_inverted_transposed_chords:
            if (verbose): print(chords[chord])


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
    if (verbose): print(len(out_of_range_chords), "chords have outlying voicings. Removing from array:")
    out_of_range_chords.sort(reverse=True) # reverse the order of the invalid chords indices as need to remove from list in reverse order
    for index in out_of_range_chords:
        del chords_in_c[index]
        if (verbose): print("chord at index ", index, "deleted")
    if (verbose): print(len(chords_in_c))
    


def test_transpose_extreme_octaves(is_user_test=True):
    global chords_in_c
    for chord in chords_in_c:
        if chord['note_numbers'][0] < 16 or chord['note_numbers'][0] > 40:
            if is_user_test and (verbose): print("Error - some chords still in extreme octaves")
            return False
    if is_user_test and (verbose): print("Success - all chords in desirable range")
    return True
    
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


def write_training_data():
    training_data_pickle = pickle.dumps(training_data)
    training_data_2d_pickle = pickle.dumps(training_data_2d)
    with open('training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)
    with open('training_data_1d.pickle', 'wb') as file:
        pickle.dump(training_data_1d, file)
    with open('training_data_2d.pickle', 'wb') as file:
        pickle.dump(training_data_2d, file)


def write_training_data_csv_note_numbers():
    with open('training_data_note_numbers.csv', mode='w') as csv_file:
        fieldnames = ['label', 'notes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for chord in chords_in_c:
            label = chord['type']
            notes = "".join(map(str, chord['note_numbers']))
            writer.writerow({'label': label, 'notes': notes})
    

def write_training_data_csv_note_vectors():
    with open('training_data_note_vectors.csv', mode='w') as csv_file:
        fieldnames = ['label', 'notes']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for chord in chords_in_c:
            label = chord['type']
            notes = "".join(map(str, chord['notes']))
            writer.writerow({'label': label, 'notes': notes})

    

def main():
    user_dir = ""
    user_deviation = 0

    parser = argparse.ArgumentParser(description="Chord Scraper Tool")

    parser.add_argument('input_directory', nargs='?', type=str, help="path to input directory. Default is current directory.")
    parser.add_argument('output_directory', nargs='?', type=str, help="path to output directory. Default is current directory.")
    parser.add_argument('-v', '--verbose', action='store_const', const='verbose', help="log parsing actions and errors.")
    parser.add_argument('-m', '--meta', action='store_const', const='meta', help="output meta information about extracted data.")
    
    parser
    args = parser.parse_args()
    print(args)

    if (args.input_directory != None):
        set_in_directory(args.input_directory)
    
    if (args.output_directory != None):
        set_out_directory(args.output_directory)
    
    if(args.verbose != None):
        set_verbose(True)

    if(args.meta != None):
        set_meta(True)
    
    mine_chords_from_dir(directory)
    flatten_chords()

    if (meta):
        chords_pretty_print(chords)

    remove_invalid_chords()
    add_note_numbers()
    sort_notes()
    if (meta):
        count_num_inverted_chords()
    transpose_chords_to_key_c()
    test_transpose_to_c()
    transpose_extreme_octaves()
    test_transpose_extreme_octaves()
    create_chord_label_vectors()
    
    write_training_data_csv_note_numbers()
    
    convert_notes_to_88_key_vectors()
    convert_labels_to_numpy_arrays() #use binary number instead
    
    # write_training_data_csv_note_vectors()
    # create_training_data()
    # create_1d_training_data()
    # create_2d_training_data()
    # write_training_data()

    if (meta):
        gather_chord_type_meta_data()

main()



