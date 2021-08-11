import xml.etree.ElementTree as ET

from fifths_to_key import fifths_to_key, alter_dict



# musicxml uses 2 elements to convey key signatures: (1) <fifths> = number indicates circle of fifth
# value ie. 1 = G, -1 = F. (2) <mode> = indicates major vs minor
def get_key_signature(attri_ele):
    mode_symbol = { # + = major, "-" = minor
        "major": "+",
        "minor": "-"
    }

    key_ele = attri_ele.find('key')
    fifth = key_ele.find('fifths').text
    mode = "major" # some xml files don't have mode element, meaning mode is by default major

    try:
        mode = key_ele.find('mode').text
    except Exception:
        pass

    return [fifths_to_key[fifth], mode_symbol[mode]]
    
def get_time_signature(attri_ele):
    time_ele = attri_ele.find('time')
    beats = time_ele.find('beats').text
    note_value = time_ele.find('beat-type').text
    return [beats, note_value]

def get_divisions(attri_ele):
    return attri_ele.find('divisions').text
# here we pass in a bar, and return the chord symbols with their extensions, as well as the notes that fall on the chords beat in the bar. The latter
# note information helps to arrange the chord
def get_chord_data(bar):
    bar_number = bar.attrib['number']
    bar_data = {'bar_number': bar_number, 'chords': []}
    chords = bar.findall('harmony')
    for chord in chords:
        extensions = {}
        harmony_ele = chord
        degree_eles = harmony_ele.findall('degree') # each <degree> element contains a chord extension
        chord = convert_root_ele_to_chord_sym(harmony_ele)
        chord["extensions"] = []
        if (degree_eles):
            extensions = extract_extensions(degree_eles)
            chord["extensions"] = extensions
        
        bar_data['chords'].append(chord)
    return bar_data
    

def convert_root_ele_to_chord_sym(harmony_ele):
    root_ele = harmony_ele.find('root') # <root> element encapsulates chord symbols
    chord_root = root_ele.find('root-step').text
    root_adjust = ""
    try:
        root_adjust = root_ele.find('root-alter').text
    except Exception:
        pass
    if (root_adjust):
        chord_root = chord_root + alter_dict[root_adjust]
    chord_kind = harmony_ele.find('kind').text
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
    return {'root': chord_root, 'bass_note': bass_note, 'kind': chord_kind}


def extract_extensions(degree_eles):
    extensions = []
    for degree in degree_eles:
        value = degree.find('degree-value').text
        alter = ""
        try:
            alter = degree.find('degree-alter').text
        except Exception:
            pass
        degree_type = degree.find('degree-type').text
        extensions.append({"degree": value, "alter": alter, "type": degree_type})
    return extensions


def extract_leadsheet_data(xml_tree):
    all_bars = xml_tree.find('part').findall('measure')
    attribute_element = all_bars[0].find('attributes') # contains all meta info about song (key/time signature etc)

    # get key and time signature
    key_sig = get_key_signature(attribute_element)
    time_sig = get_time_signature(attribute_element)
    divisions = get_divisions(attribute_element)
    
    song_data = {
        "key": key_sig,
        "time": time_sig,
        "divisions": divisions,
        "bars": [],
    }

    for bar in all_bars:
        song_data['bars'].append(get_chord_data(bar))
    
    return song_data
