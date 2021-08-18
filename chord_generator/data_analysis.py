import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect
import sys
import pickle
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from chord_scraper.utils_dict import note_number_to_name, unwanted_chord_tones, noteMidiDB
import copy

out_dir = "data_visualisations/"

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sns.set_context('notebook')
# sns.set_style("ticks")

df = pd.read_csv('training_data/label_note_vectors.csv')

print(df.head())
print(df.dtypes)
print(df.describe())

print("number of null entries: ", df.isnull().sum())

value_count=df['label'].value_counts()
print(value_count)

# plt.figure(figsize=(15, 8));
# countplot = sns.countplot(y='label',data = df, orient="h");
# countplot.set_yticklabels(countplot.get_yticklabels(), fontsize=14)
# countplot.set_xticklabels([0,250,500,750,1000,1250,1500,1750,2000], fontsize=14)
# countplot.set_xlabel("", fontsize=16)
# countplot.set_ylabel("", fontsize=16)

# for bar in countplot.patches:
#     countplot.annotate('{:}'.format(bar.get_height()), (bar.get_x()+0.15, bar.get_height()+1), fontsize=15)
# plt.show()


# plt.savefig(out_dir + "chords_count_unfiltered.png")


# converts notes to np arrays
for index, row in df.iterrows():
    row['notes'] = [int(d) for d in str(row['notes'])]
    row['notes'] = np.array(row['notes'])

# remove chord types with less than 600 chord count
df_filtered = df.groupby('label').filter(lambda x: len(x) >= 600)

value_count_filtered=df_filtered['label'].value_counts()
print(value_count_filtered)

# plt.figure(figsize=(50,5));
# sns.countplot(x='label',data = df_filtered);
# plt.savefig("chords_count_filtered.png")

filtered_labels = df_filtered['label'].unique()



for label in filtered_labels:
    print(label)

dominant_chords = df_filtered[(df_filtered.label=="dominant")]
minor_seventh_chords = df_filtered[(df_filtered.label=="minor-seventh")]
major_chords = df_filtered[(df_filtered.label=="major")]
major_seventh_chords = df_filtered[(df_filtered.label=="major-seventh")]



def create_array_of_column_values(column_name, dataframe):
    array = np.empty([dataframe[column_name].values.shape[0], 88])
    i = 0
    for value in dataframe[column_name]:
        array[i] = value
        i += 1
    return array


def stack_plot_chords(dataframe):
    array = create_array_of_column_values("notes", dataframe)
    notes = np.arange(start=1, stop=89, step=1)
    plt.stackplot(notes, array[0:50], colors=["#9b59b6"])
    plt.xlabel('note numbers')
    plt.ylabel('occurences')
    plt.show()



# stack_plot_chords(dominant_chords)
# stack_plot_chords(minor_seventh_chords)
# stack_plot_chords(major_chords)
# stack_plot_chords(major_seventh_chords)


def plot_filtered_chords(filtered_df):
    filtered_labels = df_filtered['label'].unique()

dominant_chords_array = create_array_of_column_values("notes", dominant_chords)
minor_seventh_chords_array = create_array_of_column_values("notes", minor_seventh_chords)
major_chords_array = create_array_of_column_values("notes", major_chords)
major_seventh_chords_array = create_array_of_column_values("notes", major_seventh_chords)
notes = np.arange(start=1, stop=89, step=1)



# plot as graph
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].stackplot(notes, dominant_chords_array)
# axs[0, 0].set_title('Dominant 7th')
# axs[0, 1].stackplot(notes, minor_seventh_chords_array)
# axs[0, 1].set_title('Minor 7th')
# axs[1, 0].stackplot(notes, major_chords_array)
# axs[1, 0].set_title('Major 7th')
# axs[1, 1].stackplot(notes, major_seventh_chords_array)
# axs[1, 1].set_title('Major 7th')

# for ax in axs.flat:
#     ax.set(xlabel='note-numbers', ylabel='num occurences')
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
# # plt.show()


# plot as bar chart

# Iterate through each octave (set of 12 notes) in the array and remove notes found in the chords unwanted tones list
# We start at the first octave note 3, and end at note 86.
def remove_unwanted_chord_tones(chord_array, label):
    cleaned_chord_array = copy.deepcopy(chord_array)
    for chord in cleaned_chord_array:
        for idx, note in enumerate(chord):
            if chord[idx] == 1 and note_as_integer_notation(idx) in unwanted_chord_tones[label]:
                chord[idx] = 0
    return cleaned_chord_array

# takes a note in its vector representation (integer between 0-87)
# and converts it to its integer_notation representation (number between 0-11)
def note_as_integer_notation(note):
    return (note + 9) % 12

def plot_chords_stacked_bar_chart(chords_array, title):
    plt.figure(figsize=(25,10))
    barWidth = 0.7
    cum_size = np.zeros(88)
    for idx, chord in enumerate(chords_array):
        plt.bar(notes, chords_array[idx], bottom=cum_size, width=barWidth, color="#4eabb7")
        cum_size += chords_array[idx]
    x_ticks = list(note_number_to_name.values())
    x_ticks.reverse()

    # Custom X axis
    plt.xticks(notes, x_ticks, fontsize=12, rotation=90)
    plt.yticks(fontsize=12)

    plt.xlabel("Notes", fontsize=20)
    plt.ylabel("Occurences", fontsize=20)
    plt.title(title)
    # plt.show()
    plt.savefig(out_dir + title.lower().replace(" ", "_") + ".png")


# We do not want to generate chords that will have notes that are higher in pitch than their accompanying melody notes
# Removes all notes above G4
def reduce_high_pitch_notes(chord_array):
    chord_array_reduced = copy.deepcopy(chord_array)
    for chord in chord_array_reduced:
        g4 = noteMidiDB['G4'] - 1 # make 0 indexed
        chord[g4:87] = 0
    return chord_array_reduced

def count_total_notes_in_chord_type(chord_array):
    count = 0
    for chord in chord_array:
        for note in chord:
            if note == 1:
                count +=1
    return count

def count_total_unwanted_notes_in_chord_type(chord_array, label):
    count = 0
    for chord in chord_array:
        for idx, note in enumerate(chord):
            if chord[idx] == 1 and note_as_integer_notation(idx) in unwanted_chord_tones[label]:
                count += 1
    return count

total_notes_dominant = count_total_notes_in_chord_type(dominant_chords_array)
unwanted_note_count_dominant = count_total_unwanted_notes_in_chord_type(dominant_chords_array, "dominant")
total_notes_minor_seventh = count_total_notes_in_chord_type(minor_seventh_chords_array)
unwanted_note_count_minor_seventh = count_total_unwanted_notes_in_chord_type(minor_seventh_chords_array, "minor-seventh")
total_notes_major_seventh = count_total_notes_in_chord_type(major_seventh_chords_array)
unwanted_note_count_major_seventh = count_total_unwanted_notes_in_chord_type(major_seventh_chords_array, "major-seventh")
total_notes_major = count_total_notes_in_chord_type(major_chords_array)
unwanted_note_count_major = count_total_unwanted_notes_in_chord_type(major_chords_array, "major")

def calculate_margin_of_error(chord_type, total_notes, total_unwanted_notes):
    accuracy = 0
    if (total_unwanted_notes == 0): accuracy = 1
    else:
        accuracy = 1 - (total_unwanted_notes/total_notes)
    print("-----------")
    print(chord_type, ":")
    print("Total notes:", total_notes)
    print("Total unwanted notes:", total_unwanted_notes)
    print("Accuracy:", accuracy)
    print("-----------")

calculate_margin_of_error("dominant", total_notes_dominant, unwanted_note_count_dominant)
calculate_margin_of_error("minor_seventh", total_notes_minor_seventh, unwanted_note_count_minor_seventh)
calculate_margin_of_error("major_seventh", total_notes_major_seventh, unwanted_note_count_major_seventh)
calculate_margin_of_error("major", total_notes_major, unwanted_note_count_major)

plot_chords_stacked_bar_chart(major_seventh_chords_array, "Major Seventh Chords Uncleaned")
plot_chords_stacked_bar_chart(minor_seventh_chords_array, "Minor Seventh Chords Uncleaned")
plot_chords_stacked_bar_chart(major_chords_array, "Major Chords Uncleaned")
plot_chords_stacked_bar_chart(dominant_chords_array, "Dominant Seventh Chords Uncleaned")

major_seventh_chords_array_cleaned = remove_unwanted_chord_tones(major_seventh_chords_array, "major-seventh")
minor_seventh_chords_array_cleaned = remove_unwanted_chord_tones(minor_seventh_chords_array, "minor-seventh")
major_chords_array_cleaned = remove_unwanted_chord_tones(major_chords_array, "major")
dominant_chords_array_cleaned = remove_unwanted_chord_tones(dominant_chords_array, "dominant")

# plot_chords_stacked_bar_chart(major_seventh_chords_array_cleaned, "Major Seventh Chords Cleaned")
# plot_chords_stacked_bar_chart(minor_seventh_chords_array_cleaned, "Minor Seventh Chords Cleaned")
# plot_chords_stacked_bar_chart(major_chords_array_cleaned, "Major Chords Cleaned")
# plot_chords_stacked_bar_chart(dominant_chords_array_cleaned, "Dominant Seventh Chords Cleaned")

major_seventh_chords_array_cleaned_reduced = reduce_high_pitch_notes(major_seventh_chords_array_cleaned[0:818])
minor_seventh_chords_array_cleaned_reduced = reduce_high_pitch_notes(minor_seventh_chords_array_cleaned[0:818])
major_chords_array_cleaned_reduced = reduce_high_pitch_notes(major_chords_array_cleaned[0:818])
dominant_chords_array_cleaned_reduced = reduce_high_pitch_notes(dominant_chords_array_cleaned[0:818])

# plot_chords_stacked_bar_chart(major_seventh_chords_array_cleaned_reduced, "Major Seventh Chords Cleaned and Reduced")
# plot_chords_stacked_bar_chart(minor_seventh_chords_array_cleaned_reduced, "Minor Seventh Chords Cleaned and Reduced")
# plot_chords_stacked_bar_chart(major_chords_array_cleaned_reduced, "Major Chords Cleaned and Reduced")
# plot_chords_stacked_bar_chart(dominant_chords_array_cleaned_reduced, "Dominant Seventh Chords Cleaned and Reduced")

print(len(major_chords_array_cleaned_reduced))
print(len(major_seventh_chords_array_cleaned_reduced))
print(len(minor_seventh_chords_array_cleaned_reduced))
print(len(dominant_chords_array_cleaned_reduced))


# loop through each chord matrix, and count the number of "major" and "minor" notes
def count_num_of_major_and_minor_notes(matrix):
    #sum all 7 octave vectors together
    summed_octaves = np.sum(matrix, axis=0)

    # notes that are exclusive to major chords = [4,6,11] (integer notation)
    major_notes_count = summed_octaves[4] + summed_octaves[6] + summed_octaves[11]
    
    # notes that are exclusive to minor chords = [3,5,10] (integer notation)
    minor_notes_count = summed_octaves[3] + summed_octaves[5] + summed_octaves[10]

    
    if major_notes_count > minor_notes_count:
        major_notes_count += random.uniform(0.2,2)
        minor_notes_count -= random.uniform(-0.2,2)
    elif minor_notes_count > major_notes_count:
        major_notes_count -= random.uniform(0.2,2)
        minor_notes_count += random.uniform(-0.2,2)
    if summed_octaves[4] and summed_octaves [10] == 1:
        major_notes_count += random.uniform(0,2)
        minor_notes_count += random.uniform(0,2)
    return [major_notes_count, minor_notes_count]

def create_chord_vectors_training_data():
    labels = []
    voicings = []
    for i in range(818):
        labels.append(0)
        voicings.append(dominant_chords_array_cleaned_reduced[i])
        labels.append(1)
        voicings.append(minor_seventh_chords_array_cleaned_reduced[i])
        labels.append(2)
        voicings.append(major_seventh_chords_array_cleaned_reduced[i])
        # labels.append(3)
        # voicings.append(major_chords_array_cleaned_reduced[i])
    labels_np = np.array(labels)
    voicings_np = np.array(voicings)
    training_data = (labels_np, voicings_np)
    with open('chord_vectors_training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)


def create_scatter_plot_chord_clusters():
    dominant_notes_type_count = [[], []]
    minor_seventh_notes_type_count = [[], []]
    major_notes_type_count = [[], []]
    major_seventh_notes_type_count = [[], []]

    for i in range(len(dominant_chords_array_cleaned_reduced)):
        dom_maj, dom_min = count_num_of_major_and_minor_notes(convert_chord_vector_to_matrix(dominant_chords_array_cleaned_reduced[i]))
        dominant_notes_type_count[0].append(dom_maj)
        dominant_notes_type_count[1].append(dom_min)
        min_7_maj, min_7_min = count_num_of_major_and_minor_notes(convert_chord_vector_to_matrix(minor_seventh_chords_array_cleaned_reduced[i]))
        minor_seventh_notes_type_count[0].append(min_7_maj)
        minor_seventh_notes_type_count[1].append(min_7_min)
        maj_maj, maj_min = count_num_of_major_and_minor_notes(convert_chord_vector_to_matrix(major_chords_array_cleaned_reduced[i]))
        major_notes_type_count[0].append(maj_maj)
        major_notes_type_count[1].append(maj_min)
        maj_7_maj, maj_7_min = count_num_of_major_and_minor_notes(convert_chord_vector_to_matrix(major_seventh_chords_array_cleaned_reduced[i]))
        major_seventh_notes_type_count[0].append(maj_7_maj)
        major_seventh_notes_type_count[1].append(maj_7_min)

    plt.scatter(dominant_notes_type_count[0], dominant_notes_type_count[1], color="brown", s=2)
    plt.scatter(minor_seventh_notes_type_count[0], minor_seventh_notes_type_count[1], color="green", s=2)
    plt.scatter(major_notes_type_count[0], major_notes_type_count[1], color="red", s=2)
    plt.scatter(major_seventh_notes_type_count[0], major_seventh_notes_type_count[1], color="blue", s=2)
    plt.tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False,labelleft=False)
    plt.xlabel("Major Notes")
    plt.ylabel("Minor Notes")
    plt.legend(["dominant", "minor-seventh", "major", "major-seventh"])
    plt.show()

def create_chord_matrices_training_data():
    labels = []
    voicings = []
    for i in range(818):
        labels.append(0)
        voicings.append(convert_chord_vector_to_matrix(dominant_chords_array_cleaned_reduced[i]))
        labels.append(1)
        voicings.append(convert_chord_vector_to_matrix(minor_seventh_chords_array_cleaned_reduced[i]))
        labels.append(2)
        voicings.append(convert_chord_vector_to_matrix(major_seventh_chords_array_cleaned_reduced[i]))
        # labels.append(2)
        # voicings.append(convert_chord_vector_to_matrix(major_chords_array_cleaned_reduced[i]))
    labels_np = np.array(labels)
    voicings_np = np.array(voicings)
    training_data = (labels_np, voicings_np)
    with open('chord_matrices_training_data.pickle', 'wb') as file:
        pickle.dump(training_data, file)

def convert_chord_vector_to_matrix(vector):
    clipped_vector = np.array(copy.deepcopy(vector[3:87]))
    matrix = np.reshape(clipped_vector, (7, 12))
    return matrix
    



# create_chord_vectors_training_data()

# create_chord_matrices_training_data()



# create_scatter_plot_chord_clusters()