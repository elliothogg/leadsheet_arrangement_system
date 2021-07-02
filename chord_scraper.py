import re
import os
from note_name_to_number import noteMidiDB
from chords import stored_chords

chords = []
chords_meta_data = {}
deviation = 20 #allows notes either-side of the chord location to be included in the chord (notes next to each other in a chord are usually ofset on the x-axis)
directory = os.fsencode(".") #directory of musicXML files


def alter_note(note, step):
    if step == "1":
        return note + "#"
    if step == "-1":
        return note + "b"
    return note

def sort_notes():
    global chords
    for chord in chords:
        chord['note_numbers'].sort()

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
            if "<measure number" in line:
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
            print(filename)
            # count_degree_tags(filename) # checks if degree tags are even, essential for extensions to be gathered correctly
            get_chords_meta(filename)

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

def chords_pretty_print():
    j=0
    for chord in chords:
        print(j, chord)
        j = j + 1


def main():
    mine_chords_from_dir(directory)
    flatten_chords()
    remove_invalid_chords()
    add_note_numbers()
    sort_notes()
    print(chords)

# main()

gather_chord_type_meta_data()

