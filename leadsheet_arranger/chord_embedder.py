from chord_symbol_extractor import extract_leadsheet_data
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from note_name_to_number import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation

file_path = "leadsheet_arranger/A_Time_for_Love-Bill_Evans.musicxml"

leadsheet_data = extract_leadsheet_data(file_path)

# print(leadsheet_data)

def chords_to_array(leadsheet_data):
    chords = []
    for bar in leadsheet_data['bars']:
        for chord in bar['chords']:
            chords.append(chord)
    return chords

chords = chords_to_array(leadsheet_data)

# We ignore the root of the chord, as we are only concerned with the type. We will transpose chords back to their roots when creating arrangement
def convert_chords_to_integer_notation(chords):
    chords_int = []
    for chord in chords:
        chords_int.append({'chord': chord_label_to_integer_notation[chord['kind']], 'extensions': chord['extensions']})
    print(chords_int)
    return chords_int

def add_extensions_to_chord_integer_notation(chords_int_notation):
    chords_int_exten = []
    for chord in chords_int_notation:
        print(chord)
        chord_inc_extensions = chord['chord']
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
            # print(chord_inc_extensions, extension, note_int)

def embed_chords(chords):
    embedded_chords = []
    C4 = noteMidiDB['C4'] - 1 # needs to be 0 indexed
    for chord in chords:
        chord_integer_notation = chord_label_to_integer_notation[chord['kind']]
        print(chord_integer_notation)
        embedding_array = np.zeros(88, dtype='int8')
        for i in range(C4, C4 + 12):
            print(i)

chords = add_extensions_to_chord_integer_notation(convert_chords_to_integer_notation(chords))
