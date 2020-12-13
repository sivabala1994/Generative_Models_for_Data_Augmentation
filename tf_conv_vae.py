# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython import display
import tqdm
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import time
from model import CVAE
(train_images, train_labels), (test_images, _) = tf.keras.datasets.cifar10.load_data()

def separate_data(train_images,train_labels,label):
    if label==-1:
        return train_images
    else:
        train_list=[]
        for item in range(50000):
            if int(train_labels[item]) ==label:
                train_list.append(train_images[item])
        return np.asarray(train_list)
       
            
        
def preprocess_images(images):
  images = images.reshape((images.shape[0], 32, 32, 3)) / 255.
  return images.astype('float32')
  # return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images=separate_data(train_images,train_labels,-1)
# train_images=np.asarray(train_list)

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
# test_images=train_images

train_size = train_images.shape[0]
batch_size = 32
test_size = test_images.shape[0]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

optimizer = tf.keras.optimizers.Adam(1e-4)
# optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x,beta=1):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  reconstruction_loss = -tf.reduce_sum(tf.keras.losses.MSE(x_logit,x), axis=(1,2)) 
  logpz = beta * log_normal_pdf(z, 0., 1.)
  logqz_x =beta * log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(reconstruction_loss + logpz - logqz_x),tf.reduce_mean(logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss,kl = compute_loss(model, x,0.05)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return {
        "loss": loss,
        "kl_loss": kl,
    }


def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(3, 3))

  for i in range(predictions.shape[0]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(predictions[i, :, :, :],interpolation='nearest')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
epochs = 50
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 16
num_examples_to_generate = 9

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
loss = tf.keras.metrics.Mean()
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]


for i in range(num_examples_to_generate):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_sample[i, :, :, :])
    plt.axis('off')

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  train_steps = train_size//batch_size
  with tqdm.tqdm(total=train_steps) as t:
      for train_x in train_dataset:
        loss_value=train_step(model, train_x, optimizer)
        end_time = time.time()
        t.set_description(
                            'epoch {:03d}  Loss: {:.3f} kl {:.3f} '.format(
                                epoch, loss_value["loss"],
                                loss_value["kl_loss"]))
        t.update(1)

      loss = tf.keras.metrics.Mean()
      for test_x in test_dataset:
        loss(compute_loss(model, test_x))
      elbo = -loss.result()
      display.clear_output(wait=False)
      print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            .format(epoch, elbo, end_time - start_time))
      generate_and_save_images(model, epoch, test_sample)
      
          
          
#save encoder and decoder separately         
model.encoder.save("models/enc_v2_"+str(epochs)+".hdf5") 
model.decoder.save("models/dec_v2_"+str(epochs)+".hdf5")
