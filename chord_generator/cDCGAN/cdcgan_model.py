import pickle
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.layers import Embedding, Concatenate, MaxPooling2D
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
gparentdir = os.path.dirname(parentdir)
sys.path.insert(0, gparentdir)
from leadsheet_arranger.chord_generator import generate_chords, test_generated_chords

# Loads the embedded Jazz-Chords dataset matrices and reshape chord voicing matrices from 7X12 to 3x12
def load_data():
    data_pickle_path = '../training_data/chord_matrices_training_data.pickle'
    with open(data_pickle_path, 'rb') as file:
        training_data = pickle.load(file)
    t_1 = reshape_3_x_12(training_data)
    # t_2 = replace_0_with_minus_1(t_1)
    return t_1

# Reshapes 7x12 chord voicing matrices to 12x12. Not used in current iteration
def reshape_12_x_12(data):
    y, x = data
    x_reshaped = []
    for voicing in x:
        reshape = voicing.copy()
        reshape.resize(12,12)
        x_reshaped.append(reshape)
    x_reshaped = np.array(x_reshaped)
    return (y, x_reshaped)

# Reshapes 7x12 chord voicing matrices to 3x12
def reshape_3_x_12(data):
    y, x = data
    x_reshaped = []
    for voicing in x:
        reshape = voicing.copy()
        reshape = reshape[1:4]
        x_reshaped.append(reshape)
    x_reshaped = np.array(x_reshaped)
    return (y, x_reshaped)

# Replaces all 0's in chord voicing matrices with -1 - in line with tanh activation of generator
def replace_0_with_minus_1(data):
    y, x = data
    replaced = x.copy()
    replaced = np.where(replaced == 0, -1, replaced)
    return (y, replaced)

# Discriminator model
def define_discriminator(in_shape=(3,12,1), n_classes=3):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Generator model
def define_generator(latent_dim, n_classes=3):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 1 * 3
    li = Dense(n_nodes)(li)
    # # reshape to additional channel
    li = Reshape((1, 3, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 3x3 image
    n_nodes = 128 * 1 * 3
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((1, 3, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 3x12
    gen = Conv2DTranspose(128, (4,4), strides=(3,4), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load real samples from the Jazz-Chords dataset
def load_real_samples():
    # load dataset
    trainy, trainX = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)

    return [X, trainy]

# # Select real samples from Jazz-Chords dataset an assign label of 1 incidcating to Discriminator that it is real
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=3):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples and assign label of 0 indicating fake to discriminator
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y

# Plot discriminator and generator loss
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d1_hist, label='d-real')
    pyplot.plot(d2_hist, label='d-fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.savefig('plot_line_plot_loss.png')
    pyplot.close()

# plot perceivable metrics - accuracy and uniqueness of generated chord voicings after each iteration
def plot_chord_metrics_history(accuracy, uniqueness):
    dom_acc, min_7_acc, maj_7_acc = accuracy
    dom_uniq, min_7_uniq, maj_7_uniq = uniqueness
    # plot chord accuracy
    pyplot.subplot(2, 1, 1)
    pyplot.plot(dom_acc, label='dominant')
    pyplot.plot(min_7_acc, label='minor-seventh')
    pyplot.plot(maj_7_acc, label='major-seventh')
    pyplot.legend()
    pyplot.ylabel("Accuracy")
    # plot chord uniqueness
    pyplot.subplot(2, 1, 2)
    pyplot.plot(dom_uniq, label='dominant')
    pyplot.plot(min_7_uniq, label='minor-seventh')
    pyplot.plot(maj_7_uniq, label='major-seventh')
    pyplot.legend()
    pyplot.ylabel("Uniqueness")
    pyplot.xlabel("Epochs")
    # save plot to file
    pyplot.savefig('chord_metrics_plot.png')
    pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # lists for storing model metrics
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # list for storing generated chord metrics
    chords_acc = [[],[],[]]
    chords_uniq = [[],[],[]]
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, d_acc1 = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        dom_acc, min_7_acc, maj_7_acc, dom_uniq, min_7_uniq, maj_7_uniq = test_generated_chords(generate_chords(g_model))
        chords_acc[0].append(dom_acc); chords_acc[1].append(min_7_acc); chords_acc[2].append(maj_7_acc)
        chords_uniq[0].append(dom_uniq); chords_uniq[1].append(min_7_uniq); chords_uniq[2].append(maj_7_uniq)
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    plot_chord_metrics_history(chords_acc, chords_uniq)
    #save the generator model
    g_model.save('generator.h5')


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
d_model.summary()
tf.keras.utils.plot_model(d_model, "d_model.png", show_shapes=True, show_layer_names=True)
# create the generator
g_model = define_generator(latent_dim)
g_model.summary()
tf.keras.utils.plot_model(g_model, "gen_model.png", show_shapes=True, show_layer_names=True)
# create the gan
gan_model = define_gan(g_model, d_model)
# load chord voicings
dataset = load_real_samples()

# train model
train(g_model, d_model, gan_model, dataset, latent_dim)