from keras.layers import Input, LSTM, Dense, Lambda, LeakyReLU, Conv2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import Model, Sequential
from utils import get_slot_range

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

def discriminator_model_v6(dunits=[10,10,10,10],lunits=[10,10,10,10], n_streets=2,n_weeks=7,n_steps=1,n_features=1,lr=0.0002, b1=0.5, interval=30,minv=0.0, maxv=0.0):
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  minval = -0.05+minv
  maxval = 0.05+maxv
  init = initializers.RandomUniform(minval=minval, maxval=maxval)
  
  visible = Input(shape=(n_steps,n_features))


  lbda1 = Lambda(lambda x: x[:,:n_streets,:])(visible)
  lstm1 = LSTM(lunits[0],activation='relu', kernel_initializer=init, return_sequences=False, name='lstm_1')(lbda1)
  # dense1 = Dense(dunits[0],activation='relu',name='dense1')(lstm_1)
  # denseout1 = Dense(1,activation='relu',name='denseo1')(dense1)

  # second discrete variable (weekday)
  lbda2  = Lambda(lambda x: x[:,n_streets:n_streets+n_weeks,:])(visible)
  lstm2 = LSTM(lunits[1],activation='relu', kernel_initializer=init, return_sequences=False, name='lstm_2')(lbda2)
  # dense2 = Dense(dunits[1],activation='relu',name='dense2')(lstm2)
  # denseout2 = Dense(1,activation='relu',name='denseo2')(dense2)

  # third discret variable (slot)
  lbda3  = Lambda(lambda x: x[:,n_streets+n_weeks:n_streets+n_weeks+n_slots,:])(visible)
  lstm3 = LSTM(lunits[2],activation='relu',kernel_initializer=init,return_sequences=False, name='lstm_3')(lbda3)
  # dense3 = Dense(dunits[2],activation='relu',name='dense3')(lstm3)
  # denseout3 = Dense(1,activation='relu',name='denseo3')(dense3)
  
  lbda4 = Lambda(lambda x: x[:,-1:,:])(visible)
  lstm4 = LSTM(lunits[3],activation='relu',kernel_initializer=init,return_sequences=False, name='lstm_4')(lbda4)
  # dense4 = Dense(dunits[3],activation='relu',name='dense4')(lstm4)
  # denseout4 = Dense(1,activation='relu',name='denseo4')(dense4)

  merge = concatenate([lstm1, lstm2, lstm3, lstm4], axis=1)
  out = Dense(1,kernel_initializer=init,activation='relu') (merge)

  model = Model(inputs=visible,outputs=out)
  opt = Adam(lr=lr,beta_1=b1)
  #opt = Adam()
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
  return model


def define_discriminator_v0(in_shape=(48,4,1)):
  """
  Base discriminator using CNN
  """
  model = Sequential()
  model.add(Conv2D(32,(3,3), strides=(2,2),padding="same", input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(32,(3,3), strides=(2,2),padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation="sigmoid"))
  opt = Adam(lr=0.0001, beta_1=0.5)
  model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
  return model

def define_discriminator_v1(in_shape=(48,4,1)):
  """
    Base discriminator using CNN
    - Filter (2,2)
  """
  model = Sequential()
  model.add(Conv2D(32,(2,2), strides=(2,2),padding="same", input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(32,(2,2), strides=(2,2),padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation="sigmoid"))
  opt = Adam(lr=0.0001, beta_1=0.5)
  model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
  return model

def define_discriminator_v1(in_shape=(48,4,1)):
  """
    Base discriminator using CNN
    - strides (2,1)
  """
  model = Sequential()
  model.add(Conv2D(32,(2,2), strides=(2,1),padding="same", input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(32,(2,2), strides=(2,1),padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation="sigmoid"))
  opt = Adam(lr=0.0001, beta_1=0.5)
  model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
  return model

def define_discriminator_v2(in_shape=(48,4,1)):
  """
    Base discriminator using CNN
    - filter  (2,2)
    - strides (2,1)
  """
  model = Sequential()
  model.add(Conv2D(32,(2,2), strides=(2,1),padding="same", input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(32,(2,2), strides=(2,1),padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation="sigmoid"))
  opt = Adam(lr=0.0001, beta_1=0.5)
  model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
  return model