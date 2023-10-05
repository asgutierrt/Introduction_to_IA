from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense

class AutoEncoders(Model):
  def __init__(self, output_dim, layers_dim):
    super().__init__()
    self.encoder = Sequential([Dense(i, activation="relu") for i in layers_dim])
    self.decoder = Sequential([Dense(i, activation="relu") for i in layers_dim[::-1][1:]+[output_dim]])

  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)    
    return decoded

def encode(X, **kwargs):
  encoding_layers_dims=kwargs['encoding_layers_dims'] if 'encoding_layers_dims' in kwargs else [14,20]
  auto_encoder = AutoEncoders(len(X[0]),encoding_layers_dims)
  auto_encoder.compile(loss='mae',metrics=['mae'],optimizer='adam')
  auto_encoder.fit(X, X, epochs=15, batch_size=32, verbose=0) 
  if 'verbose' in kwargs and kwargs['verbose']:
    print(auto_encoder.encoder.summary())
    print('other:')
    print(auto_encoder.decoder.summary())
  return auto_encoder.get_layer('sequential').predict(X)
    
