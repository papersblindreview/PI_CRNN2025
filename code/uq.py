
ae_encoder = get_ae_encoder()
horizon = 108
def get_forecast(input_rbc, horizon=horizon):
  start_f = time.time()
  input_encoder = ae_encoder.predict(np.expand_dims(input_rbc, axis=0), verbose=0)
  h, c = context_builder(input_encoder, training=False)
  
  x = sequence_generator(input_encoder[:,-1:], h, c, horizon, training=False)
  forecast = ae_decoder(x, training=False) 
  return np.asarray(forecast[0])
