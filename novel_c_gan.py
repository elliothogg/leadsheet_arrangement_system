import pickle
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Reshape, Dense, Activation, ReLU, Flatten
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as pyplot
from tensorflow.keras.utils import plot_model

# load the training data
def load_real_samples():
    with open('training_data_1d.pickle', 'rb') as file:
        training_data_2d = pickle.load(file)
    real_chords, source = training_data_2d
    return [source, real_chords]

dataset = load_real_samples()


# Random testing

# source = dataset[0][1]
# real_chords = dataset[0][5]


# source_chord = np.array([dataset[0][0]])
# real_chord = np.array([dataset[1][0]])
# concat = tf.concat([source_chord, real_chord], 0)

# print(source_chord.shape)
# print(concat)


# both the "source" chord and the real chord must have shape == notes_shape
def discriminator_model():
    # Do weights need to be initialised?

    # source chord input
    in_source = Input(shape=(88,), name="discrim_in_source")
    # chord input
    in_chord = Input(shape=(88,), name="discrim_in_chord")
    
    # concatenate source and chord on y-axis (2nd axis) (88,)(88,) --> (2, 176)
    in_source_2d = Reshape((1, 88), input_shape=(88,))(in_source)
    in_chord_2d = Reshape((1, 88), input_shape=(88,))(in_chord)
    merged = Concatenate(axis=1)([in_source_2d, in_chord_2d])
    
    # define dense layers. see here for more info on how tf dense layers deal with matrices - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    d0 = Dense(88, activation="relu")(merged)
    d1 = Dense(176, activation="relu")(d0)
    d2 = Dense(176, activation="relu")(d1)
    d3 = Dense(12, activation="relu")(d2)
    f = Flatten()(d3)
    # output layer with one neuron and sigmoid activation fuction (values between 0 - 1)
    out = Dense(1, activation="sigmoid")(f)
    
    # define inputs and outputs
    discriminator_model = Model([in_source, in_chord], out, name="Discriminator")

    # optimisation function - Adam is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # expermient with loss weights
    discriminator_model.compile(optimizer=opt, loss='binary_crossentropy')
    return discriminator_model

# generator will learn to map chords source to fake chords that maximise the loss of the discriminator
def generator_model():
    # Do weights need to be initialised?

    # source chord input
    in_source = Input(shape=(88,), name="gen_in_source")
    d0 = Dense(88, activation="relu", kernel_initializer='random_normal')(in_source)
    d1 = Dense(176, activation="relu")(d0)
    d2 = Dense(176, activation="relu")(d1)
    
    #output
    d3 = Dense(88, activation="relu")(d2)
    # bottleneck values between 0 - 1
    fake_chord_out = ReLU(max_value=1)(d3)

    generator_model = Model(in_source, fake_chord_out)
    return generator_model


# We use the gan model to train the generator via the discriminators loss (using backpropegation)
def c_gan_model(d_model, g_model):
    # make the discriminator untrainable during generator training
    d_model.trainable = False
    
    #source chord input
    in_source = Input(shape=(88,), name="c_gan_in_source")

    # feed the input chord source into the generator model
    g_out = g_model(in_source)

    # feed the same chord source into the discriminator alongside the generated fake chord
    d_out = d_model([in_source, g_out])

    # source chord as input, and [classification output from generator, Fake Chord from generator] as output
    gan_model = Model(in_source, [d_out, g_out], name="c_gan_model")

    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # not sure about loss functions or loss_weights - need to read further
    gan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return gan_model


# returns 1 source chord, 1 real chord, and a label of int=1 to indicate to the discriminator that its real
def select_real_sample(dataset):
    num_samples = dataset[0].shape[0]
    source, real_chords = dataset
    random_index = np.random.randint(0, num_samples)

    source_chord = np.array([source[random_index]])
    real_chord = np.array([real_chords[random_index]])
    y = np.array([1])
    return [source_chord, real_chord], y

def generate_fake_sample(g_model, source_chord):
    # num_samples = dataset[0].shape[0]
    # source, real_chords = dataset
    # random_index = np.random.randint(0, num_samples)

    # source_chord = source[random_index]
    # print("source chord:", source_chord.shape)
    # print("source chord:", source_chord)
    fake_chord = g_model.predict(np.array([source_chord]))
    y = np.array([0]) # telling the discriminator that the chord within concat[source, chord] is FAKE
    print("fake_chord: ",fake_chord)
    return fake_chord, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [source, real_chord], _ = select_real_sample(dataset)
    # generate a batch of fake samples
    fake_chord, _ = generate_fake_sample(g_model, source)
    # scale all pixels from [-1,1] to [0,1]
    source = (source + 1) / 2.0
    real_chord = (real_chord + 1) / 2.0
    fake_chord = (fake_chord + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(source[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(fake_chord[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(real_chord[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    source, real_chords = dataset
    # more info on how to calculate below items here - https://machinelearningmastery.com/how-to-code-the-generative-adversarial-network-training-algorithm-and-loss-functions/
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(source) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    # enumerate epochs
    for i in range(n_steps):
        # randomly select real chord
        [source, real_chord], real_label = select_real_sample(dataset)
        print(source, real_chord, real_label)
        # generate fake chord using selected real chords source
        fake_chord, fake_label = generate_fake_sample(g_model, source)

        # train the discriminator using real chord and get the loss
        d_real_loss = d_model.train_on_batch([source, real_chord], real_label)
        # train the discriminator using fake chord and get the loss
        d_fake_loss = d_model.train_on_batch([source, fake_chord], fake_label)

        # train the generator using the discriminators loss
        g_loss = gan_model.train_on_batch(source, [real_label, real_chord])

        # summarize performance
        print('>%d, d1_real_loss[%.3f] d_fake_loss[%.3f] g_loss[%.3f]' % (i+1, d_real_loss, d_fake_loss, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


# test if input and output shapes of each model are correct
def test_input_output_shape(model_name, model, data_in, expected_out_shape):
    output_array = model.predict(data_in)
    output = output_array[0]
    print("model test output:", output)
    if output.shape == expected_out_shape:
        print("Success!", model_name, "output shape was as expected")
    else: print("Error!", model_name, "expected output shape -", expected_out_shape, ". Actual -", output.shape)





#gan_model = c_gan_model(dis_model, gen_model, notes_shape)


# test generator models IO shape is correct
# select source chord to test generator model
# source_chord = np.array([dataset[0][0]])
gen_model = generator_model()
# gen_model.summary()
# plot_model(gen_model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)
# print(source_chord.shape)
# test_input_output_shape("generator model", gen_model, source_chord, (88,))


# # test discriminator models IO shape is correct
# # select source chord to test generator model
# # source_chord = np.array([dataset[0][0]])
# # real_chord = np.array([dataset[1][0]])
dis_model = discriminator_model()
# # dis_model.summary()
# # plot_model(dis_model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)
# # test_input_output_shape("Discriminator model", dis_model, [source_chord, real_chord], (1,))


# test GAN models IO shape is correct
# select source chord to test generator model
source_chord = np.array([dataset[0][0]])
c_gan_model = c_gan_model(dis_model, gen_model)
c_gan_model.summary()
plot_model(dis_model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)
test_input_output_shape("C_GAN model", c_gan_model, source_chord, (1,))


train(dis_model, gen_model, c_gan_model, dataset)


# # concatinate source and chords on 2nd dimension
# real_chords = np.expand_dims(x1, axis = 1)
# source = np.expand_dims(x2, axis = 1)
# concat = np.concatenate((source, real_chord))
# print(concat.shape)