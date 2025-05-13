import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys 
sys.path.insert(1, os.path.dirname(os.getcwd()))
from functions import *
import numpy as np 
import pickle
import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import time
import math
import h5py
import gc
import io
tf.keras.utils.set_random_seed(1)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'\nNumber of GPUs: {len(gpus)}')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# MODEL HYPERPARAMETERS
epochs = 50000
train_size = 2000
val_size = 500
kernel_size = 3
nodes = [64,64,64,64]
activation = 'linear'
batch_size = 32
lr = 0.001
min_lr = 0.0001

# LOADING TRAINING AND VALIDATION DATA
const_dict = load_constants_long() #_long
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)
ae_train, ae_val, x, z = load_ae_data(train_size, val_size, batch_size, Uf, P, T_h, T_0, offset=2)
print('\nData loaded')

# LOSS FUNCTION (MSE)
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
def my_loss(U_true, U_pred):
  losses = tf.reduce_mean(tf.math.square(U_pred-U_true), axis=[0,1,2])
  return losses[0], losses[1], losses[2], losses[3]
  
optimizer = tf.keras.optimizers.AdamW(lr)

autoencoder = build_ae(nodes, kernel_size, activation)
    
############################################

start_time = time.time()                      

# TRAIN STEP FUNCTION
## For each batch we compute balanced loss
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[4], dtype=tf.float32)])
def train_step(x_batch, U_batch, lambdas):
  with tf.GradientTape(persistent=True) as tape:
    U_pred = autoencoder(x_batch, training=True)
    losses = my_loss(U_batch, U_pred) 
    loss_u, loss_w, loss_p, loss_T = losses[0], losses[1], losses[2], losses[3] 
    loss = loss_u*lambdas[0] + loss_w*lambdas[1] + loss_p*lambdas[2] + loss_T*lambdas[3]
    
  loss_data = tf.stack([loss_u, loss_w, loss_p, loss_T], axis=0)
  gradients = tape.gradient(loss, autoencoder.trainable_variables)
  optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables)) 
   
  return loss_data

# VALIDATION STEP FUNCTION
## Tracking validation loss
@tf.function(input_signature=[tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[batch_size,256,256,4], dtype=tf.float32)])
def val_step(x_batch, U_batch):
  U_pred = autoencoder(x_batch, training=False)
  losses = my_loss(U_batch, U_pred) 
  loss_u, loss_w, loss_p, loss_T = losses[0], losses[1], losses[2], losses[3] 
  return (loss_u + loss_w + loss_p + loss_T) / 4.

# HELPER VARIABLES DURING TRAINING
learning_rate = lr
best_loss = float('inf') 
lambdas = tf.Variable(tf.ones([4], tf.float32) / 4, trainable=False)
loss_history = tf.Variable(tf.zeros([4]), dtype=tf.float32, trainable=False)

Ld = tf.Variable(tf.zeros([4]), dtype=tf.float32, trainable=False)
step_tf = tf.Variable(0., dtype=tf.float32, trainable=False) 
step_val_tf = tf.Variable(0., dtype=tf.float32, trainable=False) 
w = tf.Variable(tf.zeros([4]), dtype=tf.float32, trainable=False) 
val_loss = tf.Variable(0., dtype=tf.float32, trainable=False)

# RUN TRAINING
for epoch in range(epochs):
  
  for step, (x_batch, U_batch) in enumerate(ae_train):
    step_tf.assign(step)
    Ld_temp = train_step(x_batch, U_batch, lambdas)
    Ld.assign_add(Ld_temp)
       
  Ld.assign( Ld / (step_tf+1.) )
  Ldata = tf.math.reduce_mean(Ld)
  
  if epoch >= 1:
    w.assign(Ld / loss_history)  
    lambdas.assign( tf.nn.softmax(w) )
    
  loss_history.assign(Ld)
   
  for step_val, (x_batch_val, U_batch_val) in enumerate(ae_val):
    step_val_tf.assign(step_val)
    val_loss.assign_add( val_step(x_batch_val, U_batch_val) )

  val_loss.assign( val_loss / (step_val_tf+1.) )

  # LEARNING RATE SCHEDULER IF VALIDATION LOSS PLATEAUS
  if epoch >= 100:
    if (best_loss - val_loss) > 1e-4: 
      best_loss = val_loss
      wait = 0
    else:
      wait += 1
      
    if wait >= 20:
      new_lr = max(learning_rate * 0.8, min_lr)
      if new_lr < learning_rate:
        learning_rate = new_lr
        optimizer.learning_rate.assign(learning_rate)
        with open(f'ae_{suffix}_log.txt', 'a') as log_file:
          log_file.write(f"Reduced learning rate to {learning_rate:.4e}"+'\n')
      wait = 0
      
  log = f"Epoch {epoch+1}, Loss: {Ldata.numpy():.2e}, Val Loss: {val_loss.numpy():.2e}, "
  log2 = f"(u): {Ld[0].numpy():.2e}, (w): {Ld[1].numpy():.2e}, (p): {Ld[2].numpy():.2e}, (T): {Ld[3].numpy():.2e}"  
  if (epoch+1) % 100 == 0:
    autoencoder.save(os.getcwd() + f'/models/ae_{suffix}.keras', overwrite=True)
    print(log)
   
  Ld.assign( tf.zeros_like(Ld) )  
  val_loss.assign(tf.constant(0.))  
    
train_time = time.time()
autoencoder.save(os.getcwd() + f'/models/ae_{suffix}.keras', overwrite=True)

delta_train = train_time - start_time
delta_train_hrs = int(delta_train // (60*60))
delta_train_min = int( (delta_train % (60*60)) // 60) 
delta_train_sec = int(delta_train % 60)
print(f'\nTraining time for {epochs} epochs and {train_size} observations: {delta_train_hrs} hrs {delta_train_min} min {delta_train_sec} sec')

