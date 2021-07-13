import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint

with open('training_data.pickle', 'rb') as file:
    training_data = pickle.load(file)


# x.shape = (4831, 88)  y.shape = (4831, 88)
x, y = training_data





# concatinating notes and labels vectors (88 + 12) may ive strange results
# also need to experiment with kernal size and stride -- I think kernal should = 4 and stride = 2

# define the discriminator model
def define_discriminator(notes_shape, label_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source input (label)
    in_src_image = tf.keras.Input(shape=label_shape)
    # target input (notes)
    in_target_image = tf.keras.Input(shape=notes_shape)
    # concatenate inputs channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv1D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv1D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv1D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv1D(512, 4, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv1D(16, 4, padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out, name='discriminator')
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model





# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv1D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv1DTranspose(n_filters, 4, strides=1, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
 
# define the standalone generator model
def define_generator(notes_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = tf.keras.Input(shape=notes_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv1DTranspose(88, 2, strides=1, padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image, name='generator')
    return model


def define_gan(g_model, d_model, notes_shape, label_shape):
        # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # source input (label)
    in_src = tf.keras.Input(shape=label_shape)
    # target input (notes)
    in_target = tf.keras.Input(shape=notes_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_target)
    
    # connect the source input and generator output to the discriminator input

    
    dis_out = d_model([in_src, gen_out])


    # src image as input, generated image and classification output
    model = Model([in_target, in_src], [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model

# define input shapes
notes_shape = (1, 88)
label_shape = (1, 12)


discriminator_model = define_discriminator(notes_shape, label_shape)
# # summarize the model
# discriminator_model.summary()
# # plot the model
# tf.keras.utils.plot_model(discriminator_model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)


generator_model = define_generator(notes_shape)
# # summarize the model
generator_model.summary()
# # plot the model
# tf.keras.utils.plot_model(generator_model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)


gan_model = define_gan(generator_model, discriminator_model, notes_shape, label_shape)
# summarize the model
gan_model.summary()
# plot the model
tf.keras.utils.plot_model(gan_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)








# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)

    # retrieve selected images
    X1, X2 = trainA[ix[0]], trainB[ix[0]]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, 16, 1))
    return [X1, X2], y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    sample = np.array([samples])

    X = g_model.predict(sample)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, 16, 1))
    return X, y


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples

        label = np.array([X_realB])
        notes = np.array([X_realA])

        d_loss1 = d_model.train_on_batch([label, notes], y_real)
        # update discriminator for generated samples
        
        fake = np.array([X_fakeB])
        d_loss2 = d_model.train_on_batch([label, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch([notes, label])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
 


train(discriminator_model, generator_model, gan_model, training_data)