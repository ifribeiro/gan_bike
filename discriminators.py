from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from keras.models import Model


def discriminator_model(n_steps=1,n_features=1):
  """
  Simplest discriminator possible
  """
  visible = Input(shape=(n_steps,n_features))
  lstm_1 = LSTM(50,activation='relu',return_sequences=True, name='lstm_1') (visible)
  # maybe add dropout
  #lstm_2 = LSTM(20,activation='relu',rname='lstm_2') (lstm_1)
  # maybe add dropout
  out    = Dense(1,activation='sigmoid') (lstm_1)
  model = Model(inputs=visible,outputs=out)
  opt = Adam()
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model

def discriminator_model_v1(n_steps=1,n_features=1):
  """
  Simplest discriminator possible
  """
  visible = Input(shape=(n_steps,n_features))
  lstm_1 = LSTM(70,activation='relu',return_sequences=True, name='lstm_1') (visible)
  # maybe add dropout
  #lstm_2 = LSTM(20,activation='relu',rname='lstm_2') (lstm_1)
  # maybe add dropout
  out    = Dense(1,activation='sigmoid') (lstm_1)
  model = Model(inputs=visible,outputs=out)
  opt = Adam()
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model


def discriminator_model_v2(n_hiddens=20, n_steps=1,n_features=1,lr=0.0002, b1=0.5):
  visible = Input(shape=(n_steps,n_features))
  lstm_1 = LSTM(n_hiddens,activation='relu',return_sequences=False, name='lstm_1') (visible)
  #drop_1 = Dropout(0.2) (lstm_1)
  # maybe add dropout
  #lstm_2 = LSTM(20,activation='relu',return_sequences=True, name='lstm_2') (drop_1)
  # maybe add dropout
  out   = Dense(1,activation='sigmoid') (lstm_1)
  model = Model(inputs=visible,outputs=out)
  opt = Adam(lr=0.0001,beta_1=0.5)
  #opt = Adam()
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model


def discriminator_model_v3(n_hiddens=20, n_steps=1,n_features=1,lr=0.0002, b1=0.5):
  visible = Input(shape=(n_steps,n_features))
  # dense1 = Dense(n_hiddens,activation='relu',name='dense1')(visible)
  lstm_1 = LSTM(n_hiddens,activation='relu',return_sequences=False, name='lstm_1')(visible)
  #drop_1 = Dropout(0.2) (lstm_1)
  # maybe add dropout
  #lstm_2 = LSTM(20,activation='relu',return_sequences=True, name='lstm_2') (drop_1)
  # maybe add dropout
  out   = Dense(1,activation='sigmoid') (lstm_1)
  model = Model(inputs=visible,outputs=out)
  opt = Adam(lr=lr,beta_1=b1)
  #opt = Adam()
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model