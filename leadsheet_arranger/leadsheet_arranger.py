import copy
import os
import inspect
import sys
import xml.etree.ElementTree as ET
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chord_scraper.utils_dict import noteMidiDB, chord_label_to_integer_notation, extensions_to_integer_notation, key_sig_table, note_number_to_xml_flat, note_number_to_xml_sharp
from leadsheet_arranger.data_extractor import extract_leadsheet_data
from leadsheet_arranger.chord_embedder import embed_chords
from leadsheet_arranger.chord_generator import chord_generator

# This executes the main method of the Leadsheet Arrangement System (LSAS). All other subsystems are called within this script.
# It takes as input a folder containining MusicXML lead sheets, and output a folder containing MusicXML full arrangements of those leadsheets
# For further information, see report section 6


in_dir = "./leadsheets/"
out_dir = "./arranged_pieces/"
verbose = False

# Create mock chords in place of generated chords - used to test functions
def mock_generated_chords():
    mock_chords = []
    length = 56
    i = 0
    while i < length:
        mock_chords.append([28, 44, 47, 51])
        i = i + 1
    return mock_chords


# combine generated chords with leadsheet data to create full arrangement data
def create_full_arrangement_data(generated_chords, leadsheet_data):
    gen_chords_copy = copy.deepcopy(generated_chords)
    full_arrangement_data = copy.deepcopy(leadsheet_data)
    for bar in full_arrangement_data['bars']:
        num_of_chords_in_bar = int(len(bar['chords']))
        for chord in bar['chords']:
            chord['notes'] = gen_chords_copy[0]
            gen_chords_copy.pop(0)
            transpose_chord_notes(chord['notes'], chord['root'])
            convert_note_nums_to_notes(chord['notes'], full_arrangement_data['key'])
            chord['xml'] = create_xml_bass_clef_chords(chord['notes'], full_arrangement_data['time'], full_arrangement_data['divisions'], num_of_chords_in_bar)
    if (len(gen_chords_copy) != 0):
        raise ValueError('Num of generated chords is not equal to number of chords in leadsheet')
    return full_arrangement_data

# As all generated chords are rooted in C, we need to transpose them back to their original root
def transpose_chord_notes(notes, root):
    # all chords will get transposed down apart from chords with root of C3
    C4 = noteMidiDB['C4']
    root = noteMidiDB[root + "3"]
    amount = C4 - root
    if (root == noteMidiDB['C3']): return
    for idx, note in enumerate(notes):
        notes[idx] = notes[idx] - amount

# Inserts bass clef (5 bottom lines) into lead sheet so that generated chords can be inserted
def add_bass_clef(xml_tree):
    attributes = xml_tree.find('part').find('measure').find('attributes')
    treble_clef = attributes.find('clef')
    treble_clef.set("number", "1") # assign number=1 attribute to treble clef
    bass_clef = ET.XML("<clef number=\"2\"><sign>F</sign><line>4</line></clef>")
    treble_clef_index = get_treble_clef_element_index(attributes, treble_clef)
    attributes.insert(treble_clef_index + 1, bass_clef) # insert bass clef element after treble clef element
    attributes.insert(treble_clef_index - 1, ET.XML("<staves>2</staves>")) # insert stave element indicating there are 2 clefs


# Gets index of treble clef "clef" element so that bass clef "clef" element can be inserted after it
def get_treble_clef_element_index(attri_ele, treble_ele):
    i = 0
    for child in attri_ele:
        if (child == treble_ele):
            return i
        i = i + 1
    raise ValueError('treble clef xml format not recognised')

# Converts a note represented as a note number to a note label, i.e. 16 -> C2
def convert_note_nums_to_notes(notes, key_sig):
    key_type = key_sig_table["".join(key_sig)] #sharp or flat notes used
    for idx, note in enumerate(notes):
        if (key_type == "b"):
            notes[idx] = note_number_to_xml_flat[notes[idx]]
        else:
            notes[idx] = note_number_to_xml_sharp[notes[idx]]

# Takes a list of note labels and converts them into MusicXML note elements that can be inserted into the arrangement
def create_xml_bass_clef_chords(notes, time_sig, divisions, num_chords_bar):
    xml_notes = []
    note_length_type, dotted, duration = get_note_length(divisions, time_sig, num_chords_bar)
    for idx, note in enumerate(notes):
        xml_note = ET.Element("note")
        if(idx != 0):
            chord = ET.SubElement(xml_note, "chord") # all notes after first need chord element to indicate all one chord
        pitch = ET.SubElement(xml_note, "pitch")
        step = ET.SubElement(pitch, "step")
        step.text = notes[idx]["step"]
        if (notes[idx]["alter"]):
            alter = ET.SubElement(pitch, "alter")
            alter.text = notes[idx]["alter"]
        octave = ET.SubElement(pitch, "octave")
        octave.text = notes[idx]["octave"]
        duration = ET.SubElement(xml_note, "duration")
        duration.text = duration
        voice = ET.SubElement(xml_note, "voice")
        voice.text = "5"
        note_type = ET.SubElement(xml_note, "type")
        note_type.text = note_length_type
        # accidental?
        if (dotted):
            dot = ET.SubElement(xml_note, "dot")
        stem = ET.SubElement(xml_note, "stem")
        stem.text = "down"
        staff = ET.SubElement(xml_note, "staff")
        staff.text = "2"
        xml_notes.append(xml_note)
    return xml_notes

# loop through each measure and add the xml note elements for each chord into the full arrangement xml tree
def insert_chords_to_arrangement(arrangement_data, arrangement_xml_root):
    xml_bars = arrangement_xml_root.find('part').findall('measure')
    data_bars = arrangement_data['bars']
    divisions_per_bar = int(arrangement_data['time'][0]) * int(arrangement_data['divisions'])
    if (len(xml_bars) != len(data_bars)):
        raise ValueError('Number of xml bars vs num of data bars not equal')
    
    for idx, bar in enumerate(data_bars):
        chords_index = get_chord_symbol_indeces(xml_bars[idx])
        xml_bars[idx].append(ET.XML("<backup><duration>{divisions_per_bar}</duration></backup>".format(divisions_per_bar=divisions_per_bar))) # insert after all treble clef notes to return insertion to first beat for bass clef notes to be added
        # need to interate through multiple chords per bar!!
        if data_bars[idx]['chords']:
            i = 0
            for chord in data_bars[idx]['chords']:
                for note in data_bars[idx]['chords'][i]['xml']:
                    xml_bars[idx].append(note)
                i += 1

# Find position to insert notes
def get_chord_symbol_indeces(bar_xml):
    indices = []
    for idx, child in enumerate(bar_xml):
        if bar_xml[idx].tag == "harmony":
            indices.append(idx)
    return indices 

# Uses time signature and divisions to determine how long to make the notes of each chord voicing. See report section 6.1.3 for more information
def get_note_length(divisions, time_sig, num_chords_bar):
    duration = (int(time_sig[0]) * int(divisions)) / int(num_chords_bar)
    divisions = int(divisions)
    if time_sig[1] == "8":
        divisions = divisions * 2
    note_type = ""
    dotted = False
    if duration == divisions / 2:
        note_type = "eigth"
    if duration == divisions:
        note_type = "quarter"
    elif duration == divisions * 1.5:
        note_type = "quarter"
        dotted = True
    elif duration == divisions * 2:
        note_type = "half"
    elif duration == divisions * 3:
        note_type = "half"
        dotted = True
    elif duration == divisions * 4:
        note_type = "whole"
    else:
        note_type = "quarter"
        duration = divisions
        if (verbose): print('ISSUE: There may be errors in the arrangement. Bar(s) have too many chords. There is currently limited support for many chords per bar.')

    return [note_type, dotted, duration]


# Functions used by the CLI
def set_in_directory(direc):
    global in_dir
    in_dir = direc

def set_out_directory(direc):
    global out_dir
    out_dir = direc

# Disables/enables additional printing of program messages
def set_verbose(bool):
    global verbose
    verbose = bool

# Main method. Contains a CLI, as well as calls to other subsystems.
def main():
    #CLI 
    parser = argparse.ArgumentParser(description="Leadsheet Arrangement System (LSAS)")
    parser.add_argument('input_directory', nargs='?', type=str, help=("path to input directory. Default is " + in_dir))
    parser.add_argument('output_directory', nargs='?', type=str, help=("path to output directory. Default is " + out_dir))
    parser.add_argument('-v', '--verbose', action='store_const', const='verbose', help="log parsing actions and errors.")
    args = parser.parse_args()
    if (args.input_directory != None):
        set_in_directory(args.input_directory)
    
    if (args.output_directory != None):
        set_out_directory(args.output_directory)
    
    if(args.verbose != None):
        set_verbose(True)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Loop through each file in input directory and perform lead sheet arrangement on those ending in .musicxml or .xml
    for file in os.listdir(in_dir):
        filename = os.fsdecode(file)
        num_leadsheets_arranged = 0
        if filename.endswith(".musicxml") or filename.endswith(".xml"):
            # convert xml file to element tree
            tree = ET.parse(in_dir + filename)
            root = tree.getroot()

            leadsheet_data = extract_leadsheet_data(root)

            chord_labels = embed_chords(leadsheet_data)

            # all chords are generated with C as the root, so need transposing
            generated_chords = chord_generator(chord_labels, verbose)
            # mock_chords = mock_generated_chords() # mock chords can be used for testing

            full_arrangement_data = create_full_arrangement_data(generated_chords, leadsheet_data)


            out_name = filename.split("/")
            out_name = out_name[len(out_name)-1]

            add_bass_clef(root)
            insert_chords_to_arrangement(full_arrangement_data, root)
            tree.write(out_dir + out_name, encoding="UTF-8", xml_declaration=True)
            print("Lead sheet has been sucesfully arranged. Please find it here: " + out_dir + out_name)
            num_leadsheets_arranged += 1
    if num_leadsheets_arranged == 0: print("\nCould not find any MusicXML leadsheets in input directory")

main()