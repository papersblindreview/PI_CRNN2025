import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys 
sys.path.insert(1, os.path.dirname(os.getcwd()))
from functions import *
import numpy as np 
import pickle
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects
import time
import h5py
import gc
import io


gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'Number of GPUs: {len(gpus)}')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# LOADING TRAINED CAE
ae_path_model = os.getcwd() + '/ae_cv/models/ae.keras'
with open(ae_path_model, 'rb') as f:
  ae_model_specs = pickle.load(f)     


# HYPERPARAMETERS
epochs = 1500
offset = 2
train_size, val_size = 2000, 500
look_back, look_fwd = 80, 60
batch_size = 8
nodes, kernel_size = 96, 3
activation = 'tanh'
const_dict = load_constants_long()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)

############################################

# CREATE CONTEXT BUILDER
def get_context_builder(size, kernel_size, dropout):
  inputs = tf.keras.layers.Input(shape=(None, 16, 16, 64), name='Inputs')
  _, h1, c1 = ConvLSTM2D(size, kernel_size=kernel_size, return_sequences=True, return_state=True, padding='same', name='Encoder1')(inputs)
  return tf.keras.Model(inputs, [h1, c1], name='Encoder_Model')

# CREATE SEQUENCE GENERATOR
@keras.saving.register_keras_serializable()
class SequenceGenerator(tf.keras.Model):
  def __init__(self, hidden_size, kernel_size):
    super(SequenceGenerator, self).__init__(name='SequenceGenerator_Model')
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size
    self.rnn = ConvLSTM2D(hidden_size, kernel_size, return_state=True, padding='same', name='Decoder')
    self.conv = Conv2D(64, kernel_size=3, padding='same', name='ConvDecoder') 
    self.norm = LayerNormalization(name='NormDecoder')
    self.act = LeakyReLU(0.2, name='ReLUDecoder')
    self.expand_dim = Lambda(lambda x: tf.expand_dims(x, axis=1))
    
  def call(self, inputs):
    initial_input, h, c, targets, teach_prob = inputs
    t_switch = tf.cast(targets.shape[1], tf.float32) * teach_prob
    outputs_ta = []
    input_at_t = initial_input

    for t in range(targets.shape[1]):
      condition = tf.less(tf.cast(t, tf.float32), t_switch)
      dec_o, h, c = self.rnn(input_at_t, initial_state=[h,c])
      output = self.act(self.conv(dec_o)) #self.act(self.norm(self.conv(dec_o)))
      outputs_ta.append(output)
      input_at_t = tf.cond(condition, lambda: self.expand_dim(output), lambda: targets[:,t:t+1])
    
    return tf.stack(outputs_ta, axis=1)

  def get_config(self):    
    config = super().get_config()
    config.update({"hidden_size": self.hidden_size, "kernel_size": self.kernel_size})
    return config
    
  @classmethod
  def from_config(cls, config):
    return cls(**config)

# LOAD TRAINED SPATIAL DECODER (the input sequences are already reduced)
def get_ae_decoder(ae_path_model=ae_path_model):
  _, dec_layers = get_ae_layers(ae_path_model)
  inputs = Input(shape=(None,16,16,64), name='inputs')
  x = inputs
  for l in dec_layers: x = TimeDistributed(l, name=l.name)(x)
  return tf.keras.Model(inputs, x, name='AE_Decoder')
  

tf.keras.utils.set_random_seed(1)

# LOAD DATA AND dx, dz, dt FOR DERIVATIVES
data_train, data_val, x, z = load_lstm_data(train_size, val_size, look_back, look_fwd, stride, Uf, P, T_h, T_0, ae_path_model, offset)               
dx_np, dz_np, dt_np = get_grads(x, z, const_dict, Uf, offset)

dx = tf.constant(dx_np, tf.float32)
dz = tf.constant(dz_np, tf.float32)
dt = tf.constant(np.array(dt_np).reshape(1,), tf.float32)
print('Data loaded.\n')

# HELPER FUNCTIONS TO COMPUTE DERIVATIVES
@tf.function(input_signature=[tf.TensorSpec(shape=[bsize,look_fwd,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[1], dtype=tf.float32)]) 
def DT_tf(var, dt):
  ddt1 = (var[...,1:2,:,:,:] - var[...,:1,:,:,:]) / dt
  ddt = (var[...,2:,:,:,:] - var[...,:-2,:,:,:]) / (2*dt)
  ddt2 = (var[...,-2:-1,:,:,:] - var[...,-1:,:,:,:]) / (-dt)
  ddt = tf.concat([ddt1,ddt,ddt2], axis=-4)
  return ddt

@tf.function(input_signature=[tf.TensorSpec(shape=[bsize,look_fwd,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[256], dtype=tf.float32)]) 
def DX_tf(var, dx):
  dx = tf.reshape(dx, [1,1,tf.shape(dx)[0],1,1])
  ddx1 = var[...,1:2,:,:] - var[...,:1,:,:]
  ddx = var[...,2:,:,:] - var[...,:-2,:,:]
  ddx2 = var[...,-2:-1,:,:] - var[...,-1:,:,:]
  ddx = tf.concat([ddx1, ddx, ddx2], axis=-3)
  return ddx / dx 

@tf.function(input_signature=[tf.TensorSpec(shape=[bsize,look_fwd,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[256], dtype=tf.float32)])  
def DZ_tf(var, dz):
  dz = tf.reshape(dz, [1,1,1,tf.shape(dz)[0],1])
  ddz1 = var[...,:,1:2,:] - var[...,:,:1,:]
  ddz = var[...,:,2:,:] - var[...,:,:-2,:]
  ddz2 = var[...,:,-2:-1,:] - var[...,:,-1:,:]
  ddz = tf.concat([ddz1,ddz,ddz2], axis=-2)
  return ddz / dz 

PHYSICS LOSS WRT MASS, MOMENTUM, ENERGY CONSERVATION
@tf.function(input_signature=[tf.TensorSpec(shape=[bsize,look_fwd,256,256,4], dtype=tf.float32),
                              tf.TensorSpec(shape=[bsize,look_fwd,256,256,4], dtype=tf.float32)])
def loss_ns(U_true, U_pred): 
  
  loss_data = tf.reduce_mean(tf.math.square(U_pred-U_true), axis=[0,1,2,3])  
  Ld_u = tf.math.sqrt(loss_data[0])
  Ld_w = tf.math.sqrt(loss_data[1])
  Ld_p = tf.math.sqrt(loss_data[2])
  Ld_T = tf.math.sqrt(loss_data[3])
  
  U_pred_x = DX_tf(U_pred, dx)
  U_pred_z = DZ_tf(U_pred, dz)
  U_pred_t  = DT_tf(U_pred, dt)
  U_pred_xx = DX_tf(U_pred_x, dx)
  U_pred_zz = DZ_tf(U_pred_z, dz)  

  u, w, p, T = tf.split(U_pred, 4, axis=-1)
  u_x, w_x, p_x, T_x = tf.split(U_pred_x, 4, axis=-1)
  u_z, w_z, p_z, T_z = tf.split(U_pred_z, 4, axis=-1)
  u_t, w_t, _, T_t = tf.split(U_pred_t, 4, axis=-1)
  u_xx, w_xx, _, T_xx = tf.split(U_pred_xx, 4, axis=-1)
  u_zz, w_zz, _, T_zz = tf.split(U_pred_zz, 4, axis=-1)
  
  f_mc = u_x + w_z     
  f_u = u_t + u*u_x + w*u_z + p_x - tf.math.sqrt(Pr/Ra)*(u_xx + u_zz)
  f_w = w_t + u*w_x + w*w_z + p_z - tf.math.sqrt(Pr/Ra)*(w_xx + w_zz) - T
  f_T = T_t + u*T_x + w*T_z - (T_xx + T_zz)/tf.math.sqrt(Pr*Ra)
    
  L_mc = tf.reduce_mean(tf.math.square(f_mc))
  L_u = tf.reduce_mean(tf.math.square(f_u))
  L_w = tf.reduce_mean(tf.math.square(f_w))
  L_T = tf.reduce_mean(tf.math.square(f_T))
  
  return Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T
     
optimizer = tf.keras.optimizers.Adam(tf.constant(lstm_model_specs['lr']))


context_builder = get_context_builder(nodes[0], kernel_size, dropout)
sequence_generator = SequenceGenerator(hidden_size=nodes[0], kernel_size=kernel_size, dropout=dropout, out_size=64)
ae_decoder = get_ae_decoder()
################################################################################################# 

# VALIDATION STEP HELPER FUNCTION TO KEEP TRACK OF VALIDATION LOSS
low_dims = 256 // (2**len(ae_model_specs['nodes_enc'])) 
@tf.function(input_signature=[tf.TensorSpec(shape=[1, look_back//stride, low_dims, low_dims, 64], dtype=tf.float32),
                            tf.TensorSpec(shape=[1, look_fwd, low_dims, low_dims, 64], dtype=tf.float32),
                            tf.TensorSpec(shape=[1, look_fwd, 256, 256, 4], dtype=tf.float32),
                            tf.TensorSpec(shape=[], dtype=tf.float32)])
def val_step_pinn(x_batch, x_dec, U_batch, teach_prob):
  h, c = context_builder(x_batch, training=False)
  x = sequence_generator((x_batch[:,-1:], h, c, x_dec, teach_prob), training=False)
  U_pred = ae_decoder(x, training=False)
  
  Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T = loss_ns(U_batch, U_pred) 
  return tf.stack([Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T], axis=0)

# HELPER FUNCTION FOR TRAIN STEP FOR EACH BATCH
## Batch size of 1, gradients are accumulated to simulate larger batch size due to memory constraints
input_signature_train = [tf.TensorSpec(shape=[1, look_back//stride, low_dims, low_dims, 64], dtype=tf.float32),
                       tf.TensorSpec(shape=[1, look_fwd, low_dims, low_dims, 64], dtype=tf.float32),
                       tf.TensorSpec(shape=[1, look_fwd, 256, 256, 4], dtype=tf.float32),
                       tf.TensorSpec(shape=[8], dtype=tf.float32),
                       tf.TensorSpec(shape=[2], dtype=tf.float32),
                       tf.TensorSpec(shape=[], dtype=tf.float32)]

 
@tf.function(input_signature=input_signature_train)
def train_step_pinn(x_batch, x_dec, U_batch, l_dwa, l_g, teach_prob):
  with tf.GradientTape(persistent=True) as tape:
    h, c = encoder(x_batch, training=True)
    x = decoder((x_batch[:,-1:], h, c, x_dec, teach_prob), training=True)
    U_pred = ae_decoder(x, training=False)
    
    Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T = loss_ns(U_batch, U_pred)
    loss_data_pde = tf.stack([Ld_u, Ld_w, Ld_p, Ld_T, L_mc, L_u, L_w, L_T], axis=0)
    
    ldata = (l_dwa[0]*Ld_u + l_dwa[1]*Ld_w + l_dwa[2]*Ld_p + l_dwa[3]*Ld_T)
    lpde = (l_dwa[4]*L_mc + l_dwa[5]*L_u + l_dwa[6]*L_w + l_dwa[7]*L_T)
    loss = ldata + lpde
    
  grad_data = tape.gradient(ldata, context_builder.trainable_variables + sequence_generator.trainable_variables)
  grad_pde = tape.gradient(lpde, context_builder.trainable_variables + sequence_generator.trainable_variables)
  
  return loss_data_pde, grad_data, grad_pde


# HELPER VARIABLES FOR TRAINING
loss_history = []
l_g = tf.Variable([1. 1.], dtype=tf.float32, trainable=False) 
w = tf.constant(0.9, dtype=tf.float32)
l_dwa = tf.Variable(tf.ones([8], tf.float32) / 8, trainable=False)
 
   
start_time = time.time()
best_loss = float('inf')
patience = 20
decay_rate = 0.8

dwa_steps = tf.Variable(16, dtype=tf.int32, trainable=False)
acc_steps = tf.Variable(4, dtype=tf.int32, trainable=False)


stop_teach = look_fwd*10
warmup = 250
learning_rate = lstm_model_specs['lr']
wait = 0

grad_norm = tf.Variable(tf.zeros([2], tf.float32), trainable=False)
grad_norm_mean = tf.Variable(0., trainable=False)

grad_data_acc = [tf.Variable(tf.zeros_like(v), trainable=False) for v in context_builder.trainable_variables]
grad_data_acc += [tf.Variable(tf.zeros_like(v), trainable=False) for v in sequence_generator.trainable_variables]

grad_pde_acc = [tf.Variable(tf.zeros_like(v), trainable=False) for v in context_builder.trainable_variables]
grad_pde_acc += [tf.Variable(tf.zeros_like(v), trainable=False) for v in sequence_generator.trainable_variables]

grad_acc = [tf.Variable(tf.zeros_like(v), trainable=False) for v in context_builder.trainable_variables]
grad_acc += [tf.Variable(tf.zeros_like(v), trainable=False) for v in sequence_generator.trainable_variables]

# WE USE TEACHER FORCING WITH SCHEDULED SAMPLING
## The training starts with the model predicting one step ahead. Gradually we expose the model to longer output sequences up to the full 60
rec_prob = tf.Variable(0., trainable=False)
cur_step = tf.Variable(0, dtype=tf.int32, trainable=False)
rec_prob_val = tf.constant(1.)

# RUN TRAINING
for epoch in tf.range(1, epochs, dtype=tf.float32):
  
  rec_prob.assign( tf.minimum(1., epoch / stop_teach) )
  
  Ldata_pde = tf.zeros([8], tf.float32)
  val_loss = tf.zeros([8], tf.float32)
    
  for step, (x_train, x_dec_train, U_train) in enumerate(data_train):
    cur_step.assign(step)
    
    Ldata_pde_b, grad_data_b, grad_pde_b = train_step_pinn(x_train, x_dec_train, U_train, l_dwa, l_g, rec_prob)
    Ldata_pde += Ldata_pde_b
    
    for i in tf.range(len(grad_acc)):
      grad_data_acc[i].assign_add(grad_data_b[i] / tf.cast(acc_steps, tf.float32) )
      grad_pde_acc[i].assign_add(grad_pde_b[i] / tf.cast(acc_steps, tf.float32) )
      
    loss_history.append(Ldata_pde_b)

    # Loss balancing & gradient accumulation
    if (cur_step+1) % acc_steps == 0: 
    
      dwa_rollmean = tf.reduce_mean(loss_history[-dwa_steps:-1], axis=0)
      l_dwa.assign( tf.nn.softmax(tf.cast(loss_history[-1] / dwa_rollmean, tf.float32)) )
      
      grad_data_norm = tf.linalg.global_norm(grad_data_acc)
      grad_pde_norm = tf.linalg.global_norm(grad_pde_acc)
      
      grad_norm.assign( tf.stack([grad_data_norm, grad_pde_norm], axis=0) )
      
      l_g_hat = tf.stack([tf.constant(1.) / grad_data_norm, tf.constant(1.) / grad_pde_norm])
      l_g.assign( w*l_g + (1-w)*l_g_hat )
      l_g.assign(l_g / tf.reduce_sum(l_g))
      
      for i in tf.range(len(grad_acc)):
        grad_acc[i].assign( l_g[0]*grad_data_acc[i] + l_g[1]*grad_pde_acc[i] )
      
      max_grad_norm = tf.constant(100.0) * grad_data_norm 
      gradients, _ = tf.clip_by_global_norm(grad_acc, max_grad_norm)
      optimizer.apply_gradients(zip(gradients, context_builder.trainable_variables + sequence_generator.trainable_variables))
      
      for i in tf.range(len(grad_acc)):
        grad_acc[i].assign(tf.zeros_like(grad_pde_b[i]))
        grad_data_acc[i].assign(tf.zeros_like(grad_pde_b[i]))
        grad_pde_acc[i].assign(tf.zeros_like(grad_pde_b[i]))
        
      grad_norm.assign(tf.zeros_like(grad_norm))
      

  Ldata_pde /= (step+1)
  for step_val, (x_val, x_dec_val, U_val) in enumerate(data_val):      
    val_loss_b = val_step_pinn(x_val, x_dec_val, U_val, rec_prob_val)
    val_loss = tf.math.add_n([val_loss, val_loss_b])
    
  val_loss /= (step_val+1)
  val_data_loss = tf.reduce_mean(val_loss[:4])
  val_pde_loss = val_loss[-4:]

  # Learning rate scheduler
  if epoch >= warmup:
    if (best_loss - tf.reduce_mean(val_loss)) > 1e-3: 
      best_loss = tf.reduce_mean(val_loss)
      wait = 0
    else:
      wait += 1
      
    if wait >= patience:
      new_lr = max(learning_rate * decay_rate, lstm_model_specs['min_lr'])
      if new_lr < learning_rate:
        learning_rate = new_lr
        optimizer.learning_rate.assign(learning_rate)
        with open(f'mod_case{model_suffix}_log.txt', 'a') as log_file:
          log_file.write(f"Reduced learning rate to {learning_rate:.4e}"+'\n')
      wait = 0
   
  log1 = f"{tf.cast(epoch, tf.int32)}. ({l_g[0].numpy():.1e}, {l_g[1].numpy():.1e}), Data:{tf.reduce_mean(Ldata_pde[:4]):.2e}, "
  log2 = f"MC:{Ldata_pde[4]:.2e}, u:{Ldata_pde[5]:.2e}, w:{Ldata_pde[6]:.2e}, T:{Ldata_pde[7]:.2e}, V Data:{val_data_loss:.2e}, "
  log3 = f"V MC:{val_pde_loss[0]:.2e}, V u:{val_pde_loss[1]:.2e}, V w:{val_pde_loss[2]:.2e}, V T:{val_pde_loss[3]:.2e}"
  
  print(log1+log2+log3)
    
  if (epoch+1) % 30 == 0:  
    print(log1+log2+log3)
    context_builder.save(os.getcwd() + f'/models/context_builder.keras', overwrite=True)
    sequence_generator.save_weights(os.getcwd() + f'/models/sequence_generator.weights.h5', overwrite=True)

train_time = time.time()

delta_train = train_time - start_time
delta_train_hrs = int(delta_train // (60*60))
delta_train_min = int( (delta_train % (60*60)) // 60) 
delta_train_sec = int(delta_train % 60)

print('Done.')
print(f"\nTraining time for {epochs} epochs and {lstm_model_specs['train_size']} observations: {delta_train_hrs} hrs {delta_train_min} min {delta_train_sec} sec")







