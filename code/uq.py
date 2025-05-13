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


# THE SEQUENCE GENERATOR IN INFERENCE MODE
class SequenceGenerator(tf.keras.Model):
  def __init__(self, hidden_size, kernel_size):
    super(SequenceGenerator, self).__init__(name='Decoder_Model')
    self.hidden_size = hidden_size
    self.kernel_size = out_size
    self.rnn = ConvLSTM2D(hidden_size, kernel_size, return_state=True, padding='same', name='ConvLSTM_SG')
    self.conv = Conv2D(64, kernel_size=3, padding='same', name='Conv2D')  
    self.norm = LayerNormalization(name='Norm')
    self.act = LeakyReLU(0.2, name='ReLU')
      
  @tf.function
  def step_forward(self, input_at_t, h, c):
    output_rnn, h, c = self.rnn(input_at_t, initial_state=[h, c])
    output = self.act(self.norm(self.conv(output_rnn)))
    return output, h, c 
  
  def call(self, initial_input, h, c, horizon):
    outputs = []
    input_at_t = initial_input

    for t in range(horizon):
      output, h, c = self.step_forward(input_at_t, h, c)
      outputs.append(output)
      input_at_t = tf.expand_dims(output, axis=1)
    return tf.stack(outputs, axis=1)
      
  def get_config(self):    
    config = super().get_config()
    config.update({"hidden_size": self.hidden_size, "kernel_size": self.kernel_size})
    return config
    
  @classmethod
  def from_config(cls, config):
    return cls(**config) 

# LOAD THE FOUR COMPONENTS OF THE MODEL
ae_encoder = get_ae_encoder()
ae_decoder = get_ae_decoder()
context_builder = tf.keras.models.load_model('context_builder.keras')
sequence_generator = SequenceGenerator(hidden_size=96, kernel_size=3)
sequence_generator.load_weights('sequence_generator.weights.h5')

# LOADING DATA
const_dict = load_constants_long()
Uf, P, T_h, T_0, Pr, Ra = get_model_constants(const_dict)
train_size, val_size = 2000, 500
data_train, data_val, x, z, t = load_data(train_size, val_size, Uf, P, T_h, T_0, offset=2)
uwpT = np.concatenate((data_train, data_val), axis=0, dtype=np.float32)

look_back = 80
ensembles = 50
horizon = 108 # 2 turnover times


def get_forecast(input_rbc, horizon=horizon):
  start_f = time.time()
  input_encoder = ae_encoder.predict(np.expand_dims(input_rbc, axis=0), verbose=0)
  h, c = context_builder(input_encoder, training=False)
  
  x = sequence_generator(input_encoder[:,-1:], h, c, horizon, training=False)
  forecast = ae_decoder(x, training=False) 
  return np.asarray(forecast[0])

# the calibration forecasts begin AFTER the training window
calibration_starts = np.linspace(train_size, train_size+val_size-horizon-1, ensembles).astype('int')

residuals = []
for s in calibration_starts:
  forecast_temp = get_forecast(uwpT[(s-look_back):s])
  residuals.append(forecast_temp - uwpT[s:(s+horizon)])

residuals = np.asarray(residuals)

confidences = [0.8, 0.90, 0.95]
adj = np.array([0.97, 1.01, 0.85, 1.04]).reshape((1,1,1,4)) # found via cross validation

residuals_abs = np.abs(residuals)
     
for c in confidences:
  q = np.quantile(residuals_abs, c, axis=0)
  
  coverage = []
  for i, s in enumerate(calibration_starts):
    forecast_temp = get_forecast(uwpT[(s-look_back):s])
    temp_lo, temp_up = forecast_temp - adj*q, forecast_temp + adj*q
    cov = ( (uwpT[s:(s+horizon)] > temp_lo) & (uwpT[s:(s+horizon)] < temp_up) ).mean(axis=(0,1,2))
    coverage.append(cov)
  
  coverage = np.asarray(coverage)
  coverage_m = np.median(coverage, axis=0)
  coverage_q1 = np.quantile(coverage, 0.25, axis=0)
  coverage_q2 = np.quantile(coverage, 0.75, axis=0)
  iqr = coverage_q2 - coverage_q1
  print(f'\nCoverages ({int(100*c)}%):')
  for i, v in enumerate(['u','w','p','T']):
    print(f'{v}. {coverage_m[i]:.4f} ({iqr[i]:.4f})') 
  
















