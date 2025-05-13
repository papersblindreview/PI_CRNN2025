import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np 
import scipy.io
import tensorflow as tf
import keras
from keras import optimizers
from keras.layers import Conv2DTranspose, Conv2D, DepthwiseConv2D, ConvLSTM2D, MaxPooling2D, TimeDistributed, LeakyReLU, LayerNormalization, ReLU
from keras.layers import BatchNormalization, Dense, Flatten, Reshape, Permute, Input, Lambda, UpSampling2D, Add, Dropout, RNN, Softmax, Concatenate
from keras.layers import Attention, SpatialDropout2D, Masking, Conv3D
from keras import activations
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.saving import register_keras_serializable
import scipy
import h5py
from scipy import optimize
import gc


# LOAD COORDINATES 
def load_xzt_long():
  with h5py.File('RB_Data.mat', 'r') as mat_file:
  
    x = mat_file['xg'][:].flatten()
    z = mat_file['zg'][:].flatten()
    t = mat_file['time'][:].flatten()
    
  return x, z, t

# LOAD DNS DATA
def load_uwpT_long():
  with h5py.File('RB_Data.mat', 'r') as mat_file:
  
    u = mat_file['u'][:]
    w = mat_file['w'][:]
    p = mat_file['p'][:]
    T = mat_file['t'][:]
    
  uwpT = np.concatenate((u[...,None], w[...,None], p[...,None], T[...,None]), axis=-1)
  uwpT[...,-1] += 289.5
  uwpT = np.swapaxes(uwpT, 1, 2)
  
  x, z, t = load_xzt_long()
  return uwpT, x, z, t

# LOAD PHYSICAL CONSTANTS
def load_constants_long():
  const_dict = {}
  with h5py.File('RB_Data.mat', 'r') as mat_file:
    for key, value in mat_file.items():
      if (len(value[:].flatten()) == 1):
        try:
          const_dict[key] = np.array(value, dtype=np.float32)
        except:
          continue
      
  return const_dict

# EXTRACT CONSTANTS
def get_model_constants(const_dict):
  _, z, _ = load_xzt_long()
  try:
    Lz = z[-1] - z[0]
  except:
    Lz = const_dict['Lz']
    
  delta_T = const_dict['T_bot']-const_dict['T_top']
  Uf = tf.math.sqrt(const_dict['alpha']*const_dict['g']*Lz*delta_T)
  P = const_dict['rho_0'] * (Uf**2)
  
  Pr = const_dict['visco'][0,0] / const_dict['kappa'][0,0]
  Ra = (const_dict['alpha'][0,0]*delta_T[0,0]*const_dict['g'][0,0]*Lz) / (const_dict['visco'][0,0] * const_dict['kappa'][0,0])
   
  return Uf[0,0], P[0,0], const_dict['T_bot'][0,0], const_dict['T_top'][0,0], np.array(Pr, np.float32), np.array(Ra, np.float32)

# NONDIMENSIONALIZE
def nondim(U_pred, Uf, P, T_h, T_0):
  u, w, p, T = U_pred[...,0,tf.newaxis], U_pred[...,1,tf.newaxis], U_pred[...,2,tf.newaxis], U_pred[...,3,tf.newaxis]
  
  u, w = u/Uf, w/Uf
  p = p/P
  T = (T-T_0) / (T_h-T_0) - 0.5 
  return u, w, p, T


# HELPER FUNCTIONS TO GENERATE dx, dz, dt
def get_grads(x, z, const_dict, Uf, offset):
  try:
    Lz = z[-1] - z[0]
  except:
    Lz = const_dict['Lz'][0,0]
  
  x = x / Lz
  z = z / Lz
  
  dx = x[2:] - x[:-2] 
  dz = z[2:] - z[:-2]
  
  dx = np.concatenate((x[1:2] - x[:1], dx, x[-2:-1] - x[-1:]))
  dz = np.concatenate((z[1:2] - z[:1], dz, z[-2:-1] - z[-1:]))
  
  dt = offset * const_dict['plot_interval'][0,0] * Uf / Lz
  return tf.cast(dx, tf.float32), tf.cast(dz, tf.float32), tf.cast(dt, tf.float32) 
   
   
# LOAD TRAIN-VAL DATA SPLIT
def load_data(train_size, val_size, Uf, P, T_h, T_0, offset):
  data_dim, x, z, t = load_uwpT_long() 

  u, w, p, T = nondim(data_dim, Uf, P, T_h, T_0)
  data = np.concatenate((u, w, p, T), axis=-1)[::offset]

  data_train = data[:train_size]
  data_val = data[train_size:(train_size+val_size)]
  
  return np.array(data_train, dtype=np.float32), np.array(data_val, dtype=np.float32), x, z, t
  
# LOAD DATA FOR AE TRAINING
def load_ae_data(train_size, val_size, batch_size, Uf, P, T_h, T_0, offset): 

  data_train, data_val, x, z, t = load_data(train_size, val_size, Uf, P, T_h, T_0, offset)
  
  data_train_tf = tf.data.Dataset.from_tensor_slices((data_train, data_train))
  data_train_tf = data_train_tf.shuffle(buffer_size=train_size)
  data_train_tf = data_train_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  data_val_tf = tf.data.Dataset.from_tensor_slices((data_val, data_val))
  data_val_tf = data_val_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return data_train_tf, data_val_tf, x, z


# BUILD CAE
def build_ae(nodes_enc, kernel_size, activation):
  nodes_dec = list(reversed(nodes_enc))
  inputs = Input(shape=(256,256,4), name='inputs')
  x = inputs
     
  for i, k in enumerate(nodes_enc):
    x = Conv2D(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'ConvENC{i+1}')(x)
    x = LayerNormalization(name=f'ENCNorm{i+1}')(x)
    x = LeakyReLU(0.2, name=f'ReLUENC{i+1}')(x)
    
  for i, k in enumerate(nodes_dec):
    x = Conv2DTranspose(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'ConvDEC{len(nodes_enc)-i}')(x)
    x = LayerNormalization(name=f'DECNorm{len(nodes_enc)-i}')(x)
    x = LeakyReLU(0.2, name=f'ReLUDEC{len(nodes_enc)-i}')(x)
    
  x = Conv2D(4, kernel_size=1, activation='tanh', padding='same', name='ConvDECOut')(x)
  
  return tf.keras.Model(inputs, x, name='Autoencoder')

  
def get_ae_layers():
  autoencoder = tf.keras.models.load_model('ae.keras')

  enc_layers = []
  dec_layers = []
  for l in autoencoder.layers:
    if 'ENC' in l.name:
      enc_layers.append(l)
    elif 'DEC' in l.name:
      dec_layers.append(l)
        
  return enc_layers, dec_layers

# BUILD SPATIAL ENCODER TO REDUCE DIMENSION OF INPUT SEQUENCES TO PI-CRNN
def build_ae_encoder():
  enc_layers, _ = get_ae_layers()
  inputs = Input(shape=(None,256,256,4), name='inputs')
  x_enc = inputs
  for l in enc_layers: x_enc = TimeDistributed(l, name=l.name)(x_enc)
  return tf.keras.Model(inputs, x_enc, name='AE_Encoder')
   

# PREPARE DATA FOR PI-CRNN
def load_lstm_data(train_size, val_size, look_b, look_f, stride, Uf, P, T_h, T_0, offset):
    np.random.seed(1)
    ae_encoder = build_ae_encoder()
    data_train, data_val, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0, offset)
    data_train = tf.convert_to_tensor(data_train)
    data_val = tf.convert_to_tensor(data_val)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, look_b, 256, 256, 4], dtype=tf.float32)])
    def compress_in(x):
      return ae_encoder(x)
      
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, look_f, 256, 256, 4], dtype=tf.float32)])
    def compress_out(x):
      return ae_encoder(x)
    
    seqs = 240
    bsize = 1
    def create_dataset(starts, data):
      def generator():
        for t in starts:
          input_temp = compress_in(tf.expand_dims(data[(t-look_b):t,...], axis=0)) #data_temp = data[(t-look_b):t,...]
          input_dec_temp = compress_out(tf.expand_dims(data[t:(t+look_f),...], axis=0)) 
          output_temp = data[t:(t+look_f),...]
          yield input_temp[0], input_dec_temp[0], output_temp
              
      output_types = (tf.float32, tf.float32, tf.float32)
      output_shapes = (
          tf.TensorShape([look_b, 16, 16, 64]),
          tf.TensorShape([look_f, 16, 16, 64]),
          tf.TensorShape([look_f] + list(data.shape[1:])))
          
      return tf.data.Dataset.from_generator(generator, output_types, output_shapes)
    
    # Create datasets
    train_starts = np.random.choice(np.arange(look_b, train_size-look_f), size=seqs, replace=False)
    data_train_tf = create_dataset(train_starts, data_train)
    data_train_tf = data_train_tf.shuffle(buffer_size=seqs).batch(bsize).prefetch(tf.data.AUTOTUNE)
    
    val_starts = np.random.choice(np.arange(look_b, val_size-look_f), size=20, replace=False)
    data_val_tf = create_dataset(val_starts, data_val)
    data_val_tf = data_val_tf.batch(1).prefetch(tf.data.AUTOTUNE)
    
    return data_train_tf, data_val_tf, x, z

  
# ESN MODEL DEFINITION
@register_keras_serializable(package="Custom", name="EchoStateRNNCell")
class EchoStateRNNCell(keras.layers.Layer):
  def __init__(self, units, decay=0.1, alpha=0.5, rho=1.0, sw=1.0, seed=None,
               epsilon=None, sparseness=0.0,  activation=None, optimize=False,
               optimize_vars=None, *args, **kwargs):

    self.seed = seed
    self.units = units
    self.state_size = units
    self.sparseness = sparseness
    self.decay_ = decay
    self.alpha_ = alpha
    self.rho_ = rho
    self.sw_ = sw
    self.epsilon = epsilon
    self._activation = tf.tanh if activation is None else activation
    self.optimize = optimize
    self.optimize_vars = optimize_vars

    super(EchoStateRNNCell, self).__init__(*args, **kwargs)

  def build(self, input_shape):

    self.optimize_table = {"alpha": False, "rho": False, "decay": False, "sw": False}

    if self.optimize == True:
      for var in ["alpha", "rho", "decay", "sw"]:
        if var in self.optimize_vars:
          self.optimize_table[var] = True
        else:
          self.optimize_table[var] = False

    self.decay = tf.Variable(self.decay_, name="decay", dtype=tf.float32, trainable=self.optimize_table["decay"])
    self.alpha = tf.Variable(self.alpha_, name="alpha", dtype=tf.float32, trainable=self.optimize_table["alpha"])
    self.rho = tf.Variable(self.rho_, name="rho", dtype=tf.float32, trainable=self.optimize_table["rho"])
    self.sw = tf.Variable(self.sw_, name="sw", dtype=tf.float32, trainable=self.optimize_table["sw"])
    self.alpha_store = tf.Variable(self.alpha_, name="alpha_store", dtype=tf.float32, trainable=False) 
    self.echo_ratio = tf.Variable(1, name="echo_ratio", dtype=tf.float32, trainable=False) 
            
    self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer=keras.initializers.RandomUniform(-1, 1, seed=self.seed),
                                  name="kernel", trainable=False)

    self.recurrent_kernel_init = self.add_weight(shape=(self.units, self.units), initializer=keras.initializers.RandomNormal(seed=self.seed),
                                                name="recurrent_kernel", trainable=False)
   
    self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer=tf.zeros_initializer(), name="recurrent_kernel", trainable=False)

    self.recurrent_kernel_init.assign(self.setSparseness(self.recurrent_kernel_init))
    self.recurrent_kernel.assign(self.setAlpha(self.recurrent_kernel_init))
    self.echo_ratio.assign(self.echoStateRatio(self.recurrent_kernel))
    self.rho.assign(self.findEchoStateRho(self.recurrent_kernel*self.echo_ratio))
    
    self.built = True

  def setAlpha(self, W):
    return 0.5*(self.alpha*(W + tf.transpose(W)) + (1 - self.alpha)*(W - tf.transpose(W)))

  def setSparseness(self, W):
    mask = tf.cast(tf.random.uniform(W.shape, seed=self.seed) < (1 - self.sparseness), dtype=W.dtype)
    return W * mask

  def echoStateRatio(self, W):
    eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
    return tf.reduce_max(tf.abs(eigvals))

  def findEchoStateRho(self, W):
    target = 1.0
    eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
    x = tf.math.real(eigvals)
    y = tf.math.imag(eigvals)

    a = x**2 * self.decay**2 + y**2 * self.decay**2
    b = 2 * x * self.decay - 2 * x * self.decay**2
    c = 1 + self.decay**2 - 2 * self.decay - target**2
    sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)
    rho = tf.reduce_min(sol)
    return rho

  def clip_variables(self):
    self.decay.assign(tf.clip_by_value(self.decay, 0.00000001, 0.25))
    self.alpha.assign(tf.clip_by_value(self.alpha, 0.000001, 0.9999999))
    self.rho.assign(tf.clip_by_value(self.rho, 0.5, 1.0e100))
    self.sw.assign(tf.clip_by_value(self.sw, 0.5, 1.0e100))
    
  def call(self, inputs, states): 
    rkernel = self.setAlpha(self.recurrent_kernel_init)
    if self.alpha != self.alpha_store:
      self.clip_variables()
      self.echo_ratio.assign(self.echoStateRatio(rkernel))
      self.rho.assign(self.findEchoStateRho(rkernel*self.echo_ratio)) 
      self.alpha_store.assign(self.alpha)

    ratio = self.rho*self.echo_ratio*(1 - self.epsilon)
    prev_output = states[0]
    output = prev_output + self.decay*(tf.matmul(inputs,self.kernel * self.sw) + tf.matmul(self._activation(prev_output), rkernel*ratio) - prev_output)
    return self._activation(output), [output]
    
  def get_config(self):
    config = super(EchoStateRNNCell, self).get_config()
    config.update({
        "units": self.units,
        "decay": self.decay_,
        "alpha": self.alpha_,
        "rho": self.rho_,
        "sw": self.sw_,
        "seed": self.seed,
        "epsilon": self.epsilon,
        "sparseness": self.sparseness,
        "activation": self._activation,
        "optimize": self.optimize,
        "optimize_vars": self.optimize_vars})
    return config
    
  @classmethod
  def from_config(cls,config):
    return cls(**config)
    

# BUILD ESN
def build_ae_ESN(nodes, kernel_size, activation, ae_path_model, look_back):
  nodes_dec = list(reversed(nodes))
          
  _, dec_layers = get_ae_layers(ae_path_model)
  inputs = Input(shape=(look_back,) + (16,16,64), name='inputs')
  x = inputs

  shape = list(x.shape[2:])
  x = Reshape((x.shape[1], x.shape[2]*x.shape[3]*x.shape[4]), name='Reshape')(x)
  
  cell = EchoStateRNNCell(units=shape[-1], decay=0.1, epsilon=1e-20, alpha=0.5,
                        sparseness=0.9, optimize=False, optimize_vars=["rho", "decay", "alpha", "sw"])
                        
  x = RNN(cell, return_sequences=False, name="ESN")(x)
  
  x = Dense(np.prod(shape), name='Dense')(x)
  x = LeakyReLU(0.2, name='ReLU_ESN')(x)
  x = LayerNormalization(name='NormESN')(x)
  x = Reshape(shape)(x)
  
  x = Conv2D(dec_layers[0].output_shape[-1], kernel_size=kernel_size, padding='same', name='ConvDecoder')(x)
  x = LayerNormalization(name='NormDecoder')(x)
  x = LeakyReLU(0.2, name='ReLUDecoder')(x)
 
  for l in dec_layers: 
    x = l(x)
  
  ae_esn = tf.keras.Model(inputs, x, name='AE_ConvESN_Seq2Seq')

  return ae_esn   
    

# HELPER FUNCTION FOR THE PI-ESN 
def load_esn_data(train_size, val_size, look_b, batch_size, stride, Uf, P, T_h, T_0, ae_path_model, offset): 
  
  ae_encoder = build_ae_encoder()
  data_train, data_val, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0, offset)  
  data = np.concatenate((data_train, data_val), axis=0)
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[1, look_b, 256, 256, 4], dtype=tf.float32)])
    def compress_in(x):
      return ae_encoder(x)  
     
  input_data, output_data = [], []
  for t in range(look_b, train_size):
    data_temp = data[(t-look_b):t,...]
    input_temp = compress_in(tf.expand_dims(data_temp[::stride], axis=0))
    input_data.append(input_temp[0]) 
    output_data.append(data[t,...])
  
  input_data, output_data = np.array(input_data, np.float32), np.array(output_data, np.float32)   
  data_train_tf = tf.data.Dataset.from_tensor_slices((input_data, output_data))                                 
  data_train_tf = data_train_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  input_data, output_data = [], []
  for t in range(train_size+look_b, train_size+val_size):
    data_temp = data[(t-look_b):t,...]
    input_temp = compress_data(ae_encoder, np.expand_dims(data_temp[::stride], axis=0))
    input_data.append(input_temp[0]) 
    output_data.append(data[t,...])

  input_data, output_data = np.array(input_data, np.float32), np.array(output_data, np.float32)   
  data_val_tf = tf.data.Dataset.from_tensor_slices((input_data, output_data))                                 
  data_val_tf = data_val_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return data_train_tf, data_val_tf, x, z

