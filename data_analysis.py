import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('training_data_note_vectors.csv')

print(df.head())
print(df.dtypes)
print(df.describe())

print("number of null entries: ", df.isnull().sum())

plt.figure(figsize=(50,5));
sns.countplot(x='label',data = df);

# plt.show()
plt.savefig("chords_count_unfiltered.png")


# remove chord types with less than 600 chord count
df_filtered = df.groupby('label').filter(lambda x: len(x) >= 600)
plt.figure(figsize=(50,5));
sns.countplot(x='label',data = df_filtered);
plt.savefig("chords_count_filtered.png")