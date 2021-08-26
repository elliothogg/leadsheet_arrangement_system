# chord_generator

This contain the cDCGAN model used to generate chord voicings for a given chord label.
It also contains two data analysis and data embedding scripts that was used to embed the Jazz-Chords dataset for the models

For more information on all of this - see research paper (section 5)

## Usage

NOTE: When using your own data, the extract_chord_pairs script must be run before the chord_analysis_embedding

The trained generator model is saved at `./cDCGAN/cgan_generator.h5`
This can be imported using Keras API

You must also have graphviz installed globally for Keras model plots to work

### Retrain model

To retrain the model:

```
cd cDCGAN
python3 cdcganmodel.py
```

### Using your own dataset

If you want to retrain the model using your own dataset, ensure that you have changed the load_data function.

Your own dataset must have the shape = (`chord_labels`, `chord_voicings`):

- `chord_labels` must be single integers; the n_classes of the discriminator model must be changed accordingly

- `chord_voicings` must be 3x12 matrices (the 3 octaves from C2 - B4). If you want to include more octaves, you must adjust the models

### Data embedding & analysis

To run this script:

```
python3 chord_embedder
```