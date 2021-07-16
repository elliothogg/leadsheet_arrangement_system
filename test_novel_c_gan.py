# import each model for testing
from novel_c_gan import dis_model, gan_model, gen_model
import pickle

# import test data

# load the training data
def load_real_samples():
    with open('training_data_1d.pickle', 'rb') as file:
        training_data_2d = pickle.load(file)
    real_chords, source = training_data_2d
    print(source.shape)
    print(real_chords.shape)

    return [source, real_chords]

# test if input and output shapes of each model are correct
def test_input_output_shape(model, data_in, expected_data_out_shape):
    output = model.predict(data_in)
    print(output)


dataset = load_real_samples()
# select a source chord vector
source_chord = dataset[0][0]
real_chord = dataset[1][0]

# test discriminator model
# test_input_output_shape(dis_model, [source_chord, real_chord], (88,) )


# test generator model
test_input_output_shape(gen_model, source_chord, (88,) )
