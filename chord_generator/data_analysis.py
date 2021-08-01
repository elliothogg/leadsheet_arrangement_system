import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from chord_scraper.utils_dict import note_number_to_name, unwanted_chord_tones, noteMidiDB
import copy

df = pd.read_csv('./chord_scraper/dataset/training_data_note_vectors.csv')

print(df.head())
print(df.dtypes)
print(df.describe())

print("number of null entries: ", df.isnull().sum())

value_count=df['label'].value_counts()
print(value_count)

# plt.figure(figsize=(50,5));
# sns.countplot(x='label',data = df);

# # plt.show()
# plt.savefig("chords_count_unfiltered.png")


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

def plot_chords_bar_chart(chords_array, title):
    plt.figure(figsize=(25,10), dpi=400)
    barWidth = 0.7
    cum_size = np.zeros(88)
    for idx, chord in enumerate(chords_array):
        plt.bar(notes, chords_array[idx], bottom=cum_size, width=barWidth)
        cum_size += chords_array[idx]
    x_ticks = list(note_number_to_name.values())
    x_ticks.reverse()

    # Custom X axis
    plt.xticks(notes, x_ticks, fontsize=8, weight='bold', rotation=45)

    plt.xlabel("Notes")
    plt.ylabel("Occurences")
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_") + ".png")


# We do not want to generate chords that will have notes that are higher in pitch than their accompanying melody notes
# Removes all notes above G4
def reduce_high_pitch_notes(chord_array):
    chord_array_reduced = copy.deepcopy(chord_array)
    for chord in chord_array_reduced:
        g4 = noteMidiDB['G4'] - 1 # make 0 indexed
        chord[g4:87] = 0
    return chord_array_reduced

plot_chords_bar_chart(major_seventh_chords_array, "Major Seventh Chords Uncleaned")
plot_chords_bar_chart(minor_seventh_chords_array, "Minor Seventh Chords Uncleaned")
plot_chords_bar_chart(major_chords_array, "Major Chords Uncleaned")
plot_chords_bar_chart(dominant_chords_array, "Dominant Seventh Chords Uncleaned")

major_seventh_chords_array_cleaned = remove_unwanted_chord_tones(major_seventh_chords_array, "major-seventh")
minor_seventh_chords_array_cleaned = remove_unwanted_chord_tones(minor_seventh_chords_array, "minor-seventh")
major_chords_array_cleaned = remove_unwanted_chord_tones(major_chords_array, "major")
dominant_chords_array_cleaned = remove_unwanted_chord_tones(dominant_chords_array, "dominant")

plot_chords_bar_chart(major_seventh_chords_array_cleaned, "Major Seventh Chords Cleaned")
plot_chords_bar_chart(minor_seventh_chords_array_cleaned, "Minor Seventh Chords Cleaned")
plot_chords_bar_chart(major_chords_array_cleaned, "Major Chords Cleaned")
plot_chords_bar_chart(dominant_chords_array_cleaned, "Dominant Seventh Chords Cleaned")

major_seventh_chords_array_cleaned_reduced = reduce_high_pitch_notes(major_seventh_chords_array_cleaned)
minor_seventh_chords_array_cleaned_reduced = reduce_high_pitch_notes(minor_seventh_chords_array_cleaned)
major_chords_array_cleaned_reduced = reduce_high_pitch_notes(major_chords_array_cleaned)
dominant_chords_array_cleaned_reduced = reduce_high_pitch_notes(dominant_chords_array_cleaned)

plot_chords_bar_chart(major_seventh_chords_array_cleaned_reduced, "Major Seventh Chords Cleaned and Reduced")
plot_chords_bar_chart(minor_seventh_chords_array_cleaned_reduced, "Minor Seventh Chords Cleaned and Reduced")
plot_chords_bar_chart(major_chords_array_cleaned_reduced, "Major Chords Cleaned and Reduced")
plot_chords_bar_chart(dominant_chords_array_cleaned_reduced, "Dominant Seventh Chords Cleaned and Reduced")