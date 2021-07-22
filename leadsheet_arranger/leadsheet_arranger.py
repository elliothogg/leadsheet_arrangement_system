# main file for running arranger
# this uses pretrained model, to retrain model, see *package*

from chord_symbol_extractor import extract_leadsheet_data
from chord_embedder import embed_chords
from chord_generator import chord_generator
import copy
import os
import inspect
import sys
import xml.etree.ElementTree as ET


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from note_name_to_number import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation



file_path = "leadsheet_arranger/Someday_My_Prince_Will_Come.musicxml"

 # convert xml file to element tree
tree = ET.parse(file_path)
root = tree.getroot()

leadsheet_data = extract_leadsheet_data(root)

embedded_chords = embed_chords(leadsheet_data)

# all chords are generated with C as the root, so need transposing
generated_chords = chord_generator(embedded_chords)


def mock_generated_chords():
    mock_chords = []
    length = len(generated_chords)
    i = 0
    while i < length:
        mock_chords.append([40, 44, 47, 51])
        i = i + 1
    return mock_chords


# combine generated chords with leadsheet data to create full arrangement data
def create_full_arrangement_data(generated_chords):
    gen_chords_copy = copy.deepcopy(generated_chords)
    full_arrangement_data = copy.deepcopy(leadsheet_data)
    for bar in full_arrangement_data['bars']:
        for chord in bar['chords']:
            chord['notes'] = gen_chords_copy[0]
            gen_chords_copy.pop(0)
            transpose_chord_notes(chord['notes'], chord['root'])

    if (len(gen_chords_copy) != 0):
        raise ValueError('Num of generated chords is not equal to number of chords in leadsheet')
    return full_arrangement_data

# as all generated chords are rooted in C, we need to transpose them back to their original root
def transpose_chord_notes(notes, root):
    # all chords will get transposed down 
    C4 = noteMidiDB['C4']
    root = noteMidiDB[root + "3"]
    amount = C4 - root
    for idx, note in enumerate(notes):
        notes[idx] = notes[idx] - amount

def add_bass_clef(xml_tree):
    attributes = xml_tree.find('part').find('measure').find('attributes')
    treble_clef = attributes.find('clef')
    treble_clef.set("number", "1") # assign number=1 attribute to treble clef
    bass_clef = ET.XML("<clef number=\"2\"><sign>F</sign><line>4</line></clef>")
    treble_clef_index = get_treble_clef_element_index(attributes, treble_clef)
    attributes.insert(treble_clef_index + 1, bass_clef) # insert bass clef element after treble clef element
    attributes.insert(treble_clef_index - 1, ET.XML("<staves>2</staves>")) # insert stave element indicating there are 2 clefs


mock_chords = mock_generated_chords()

def get_treble_clef_element_index(attri_ele, treble_ele):
    i = 0
    for child in attri_ele:
        if (child == treble_ele):
            return i
        i = i + 1
    raise ValueError('treble clef xml format not recognised')

def convert_note_nums_to_notes(notes, key_sig):
    print(key_sig, notes)

def create_xml_bass_clef_chords(notes, time_sig, divisions):
    print()

full_arrangement_data = create_full_arrangement_data(mock_chords)

add_bass_clef(root)

# tree.write("test.musicxml", encoding="UTF-8", xml_declaration=True)