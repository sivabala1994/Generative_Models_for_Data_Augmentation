#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:11:27 2019

@author: siv.bala
"""
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from keras import regularizers
import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from tensorflow.keras.models import model_from_json
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.optimizers import Adam
from tf_explain.core.smoothgrad import SmoothGrad
import time
tf.config.experimental_run_functions_eagerly(True)
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_len=x_train.shape[0]
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#
##z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
#
num_classes = 10
#y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)
config_param = {
   'num_epochs':1,
   'batch_size':128,
   'optimizer':'Adam',
   'learning_rate':0.01,
   'keep_checkpoint_max':5,
   'save_checkpoints_steps':50,
   'save_summary_steps':5,
   'throttle':1,
   'checkpoint_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/checkpoints',
   'custom_model':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/custom',
   'export_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/checkpoints/export/best_exporter',
   'log_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/log',
   'tmp_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/tmp',
   'data_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/adult.pkl',
   'email':'test@test.com',
   'custom_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/custom',
   'transform_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/transform',
   'loss_function':'categorical_crossentropy'
}

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

encoder_inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 32 * 32 
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
cifar = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(cifar, epochs=30)




#def sampling(args):
#    z_mean, z_log_var = args
#    batch = tf.shape(z_mean)[0]
#    dim = tf.shape(z_mean)[1]
#    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#
#intermediate_dim=20
#latent_dim=2
#weight_decay = 1e-4
#encoder_inputs = keras.Input(shape=(32, 32, 3))
#x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
#x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Flatten()(x)
#x = layers.Dense(16, activation="relu")(x)
##h = Dense(intermediate_dim, activation='relu')(x)
#z_mean = layers.Dense(latent_dim, name="z_mean")(x)
#z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
##z = sampling()([z_mean, z_log_var])
##decoder_h = Dense(intermediate_dim, activation='relu')
##decoder_mean = Dense(, activation='sigmoid')
#x = layers.Dense(8 * 8 * 64, activation="relu")(z)
#x = layers.Reshape((8, 8, 64))(x)
#x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
#
#vae = Model(encoder_inputs, decoder_outputs)
#
## encoder, from inputs to latent space
#encoder = Model(encoder_inputs, [z_mean, z_log_var, z])
#
## generator, from latent space to reconstructed inputs
##decoder_input = Input(shape=(latent_dim,))
##_h_decoded = decoder_h(decoder_input)
##_x_decoded_mean = decoder_mean(_h_decoded)
##generator = Model(decoder_input, _x_decoded_mean)
#
##def vae_loss(encoder_inputs, decoder_outputs):
##    reconstruction_loss = tf.reduce_mean(
##            keras.losses.binary_crossentropy(encoder_inputs, decoder_outputs)
##        )
##    reconstruction_loss *= 32 * 32 *3
##    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
##    kl_loss = tf.reduce_mean(kl_loss)
##    kl_loss *= -0.5
##    total_loss = reconstruction_loss + kl_loss
##    return total_loss
#vae.compile(optimizer='adam', loss=vae_loss)
#vae.summary()
##optimizer= Adam()
#def train_step(data):
#    
#    with tf.GradientTape() as tape:
#        z_mean, z_log_var, z = encoder(data)
#        reconstruction = vae(data)
#        reconstruction_loss = tf.reduce_mean(
#            keras.losses.binary_crossentropy(data, reconstruction)
#        )
#        reconstruction_loss *= 32 * 32
#        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
#        kl_loss = tf.reduce_mean(kl_loss)
#        kl_loss *= -0.5
#        total_loss = reconstruction_loss + kl_loss
#    grads = tape.gradient(total_loss, vae.trainable_weights)
#    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
#    return {
#        "loss": total_loss,
#        "reconstruction_loss": reconstruction_loss,
#        "kl_loss": kl_loss,
#    }
##data_dir="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/train/labels.txt"
##test_dir="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/test/labels.txt"
##(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
##mnist_digits = np.concatenate([x_train, x_test], axis=0).astype("float32") / 255
###mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
##epochs = 2
##
##for epoch in range(epochs):
##    print("\nStart of epoch %d" % (epoch,))
##    start_time = time.time()
##    
##    # Iterate over the batches of the dataset.
##    train_steps = train_len//config_param['batch_size']
##    with tqdm.tqdm(total=train_steps) as t:
##        for data in generate_data(data_dir,config_param['batch_size']):
##            
##            loss_value = train_step(data)
##    
##            # Log every 200 batches.
##            t.set_description(
##                        'epoch {:03d}  Loss: {:.3f} kl {:.3f} '.format(
##                            epoch + 1, loss_value["loss"],
##                            loss_value["kl_loss"]))
##            t.update(1)
###            print("Seen so far: %d samples" % ((step + 1) * 64))
###
###    # Display metrics at the end of each epoch.
###    train_acc = train_acc_metric.result()
###    print("Training acc over epoch: %.4f" % (float(train_acc),))
###
###    # Reset training metrics at the end of each epoch
###    train_acc_metric.reset_states()
##
##
####vae.fit(, epochs=1, batch_size=128)
###vae.fit(generate_data(data_dir,config_param['batch_size']), epochs=1, batch_size=config_param['batch_size'])
#
##save to disk

def savemodel(model_path,model):
    model_json = model.to_json()
    with open(model_path+'model.json', 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(model_path+'model.h5')    


def loadmodel(model_path):
    json_file = open(model_path+"model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    
#    loaded_model.load_weights(model_path+"model.h5")
#    print("Loaded model from disk")
    return loaded_model

#loaded_model=loadmodel('/Users/siv.bala/Desktop/keras_data/scripts/')

#training
#batch_size = 64

#opt_rms = Adam()
#loaded_model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
#loaded_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])








#def modelfit(model,config_param,data_dir,test_dir):
#    opt_rms = tensorflow.keras.optimizers.Adam(lr=config_param['learning_rate'],decay=1e-6)
#    model.compile(loss=config_param['loss_function'], optimizer=opt_rms, metrics=['accuracy'])
#    
#
#    model.fit(generate_data(data_dir,config_param['batch_size']),\
#                    steps_per_epoch=x_train.shape[0] // config_param['batch_size'],epochs=config_param['num_epochs'],\
#                    verbose=1,validation_data=test_data(test_dir),callbacks=[LearningRateScheduler(lr_schedule)])



#explainer = SmoothGrad()


#modelfit(vae,config_param,data_dir,test_dir)
#grid = explainer.explain((x_train[1,:,:,:],1), model,'output', 1)
#explainer.save(grid, '.', 'smoothgrad.png')
