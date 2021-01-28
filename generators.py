
from os import name

from keras.layers import (LSTM, BatchNormalization, Conv1D, Dense, Dropout,
                          Input, Lambda, LeakyReLU, Permute, Conv2D, Conv2DTranspose,Reshape,)
from keras.layers.merge import concatenate
from keras.models import Model, Sequential

from utils import GumbelSoftmax, get_slot_range
import numpy as np

np.random.seed(0)

def generator_model_v2(dmunits=[1,1,1],n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features, n_steps))

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(visible)
  # OBS: return_sequences = True to maintaining the tensor's shape
  dense2 = Dense(n_streets*dmunits[0],activation='relu', name='dense2')(lbd1)
  lstm1 = LSTM(n_streets, return_sequences=True,name='lstm1')(dense2)
  leaky1 = LeakyReLU(alpha=0.2)(lstm1)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (leaky1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(visible)
  dense3 = Dense(n_weeks*dmunits[1],activation='relu', name='dense3')(lbd2)
  lstm2 = LSTM(n_weeks, return_sequences=True,name='lstm2')(dense3)
  leaky2 = LeakyReLU(alpha=0.2)(lstm2)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (leaky2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(visible)
  dense4 = Dense(n_slots*dmunits[2],activation='relu', name='dense4')(lbd3)
  lstm3 = LSTM(n_slots, return_sequences=True,name='lstm3')(dense4)
  leaky3 = LeakyReLU(alpha=0.2)(lstm3)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3')(leaky3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(visible)
  # leaky1 = LeakyReLU(alpha=0.2)(lstm4)
  out_4  = Dense(units=3, name='out_4') (lbd4)
  lstm4 = LSTM(units=1, return_sequences=True,name='lstm4')(out_4)
  leaky4 = LeakyReLU(alpha=0.2)(lstm4)
  merge = concatenate([out_1, out_2, out_3, leaky4])
  permute = Permute((2,1),name='permute') (merge)  
  
  # size of all discrete variables + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V3"""

def generator_model_v3(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  lstm = LSTM(n_steps, return_sequences=True, name='lstm')(visible)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(lstm)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(lstm)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(lstm)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(lstm)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  # hstack = Lambda(hstack_samples)(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=merge)

  return model

"""### V4"""

def generator_model_v4(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps*2, activation='relu', return_sequences=True,name='lstm')(visible)
  dense1 = Dense(n_steps,activation='relu',name='dense')(lstm)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(dense1)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(dense1)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(dense1)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(dense1)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V5"""

def generator_model_v5(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps, return_sequences=True,name='lstm')(visible)
  leaky1 = LeakyReLU(alpha=0.2,name='leaky1')(lstm)
  
  # lstm2 = LSTM(n_steps,activation='relu', return_sequences=True,name='lstm2')(lstm)

  #dense1 = Dense(n_steps,activation='relu',name='dense')(lstm2)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(leaky1)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(leaky1)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(leaky1)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(leaky1)
  out_4  = Dense(units=1,name='out_4') (lbd4)
  leaky2 = LeakyReLU(alpha=0.2,name='leaky2') (out_4)
  merge = concatenate([out_1, out_2, out_3, leaky2])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V6"""

def generator_model_v6(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps*2, activation='relu', return_sequences=True,name='lstm')(visible)
  dropout = Dropout(0.2) (lstm)
  dense1 = Dense(n_steps,activation='relu',name='dense')(dropout)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(dense1)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(dense1)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(dense1)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(dense1)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V7"""

def generator_model_v7(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps*2, return_sequences=True,name='lstm')(visible)
  leaky1 = LeakyReLU(alpha=0.2)(lstm)
  dense1 = Dense(n_steps,name='dense')(leaky1)
  leaky2 = LeakyReLU(alpha=0.2)(dense1)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(leaky2)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(leaky2)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(leaky2)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(leaky2)
  out_4  = Dense(units=1,name='out_4') (lbd4)
  leaky3 = LeakyReLU(alpha=0.2)(out_4)

  merge = concatenate([out_1, out_2, out_3, leaky3])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V8"""

def generator_model_v8(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))
  dense1 = Dense(n_steps*2, activation='relu', name='dense1')(visible)
  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps, activation='relu', return_sequences=True,name='lstm')(dense1)  

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(lstm)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(lstm)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(lstm)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(lstm)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V9"""

def generator_model_v9(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))
  dense1 = Dense(n_steps*2, activation='relu', name='dense1')(visible)
  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(int(n_steps*1.5), activation='relu', return_sequences=True,name='lstm')(dense1)
  dense2 = Dense(n_steps, activation='relu',name='dense2') (lstm)
  lstm2 = LSTM(n_steps,activation='relu',return_sequences=True,name='lstm2')(dense2)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(lstm2)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(lstm2)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(lstm2)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(lstm2)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model


def generator_model_v10(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))
  dense1 = Dense(n_steps*2, activation='relu', name='dense1')(visible)
  # OBS: return_sequences = True to maintaining the tensor's shape
  conv1d = Conv1D(n_steps,1,activation='relu')(dense1)
  #dense2 = Dense(n_steps,activation='relu',name='dense2')(maxpool1)

  # lstm = LSTM(int(n_steps*1.5), activation='relu', return_sequences=True,name='lstm')(dense1)
  # dense2 = Dense(n_steps, activation='relu',name='dense2') (lstm)
  # lstm2 = LSTM(n_steps,activation='relu',return_sequences=True,name='lstm2')(dense2)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(conv1d)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(conv1d)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(conv1d)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(conv1d)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

"""### V11"""

def generator_model_v11(n_streets=2, n_weeks=7, interval=30, n_features=None):
  """
  Defines the generator model
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  # OBS: return_sequences = True to maintaining the tensor's shape
  lstm = LSTM(n_steps*2, activation='relu', return_sequences=True,name='lstm')(visible)
  batch1 = BatchNormalization(momentum=0.1)(lstm)
  dense1 = Dense(n_steps,activation='relu',name='dense')(batch1)

  # first discrete variable
  lbd1   = Lambda(lambda x: x[:,:,:n_streets])(dense1)
  out_1  = GumbelSoftmax(units=n_streets,name='out_1') (lbd1)

  # second discrete variable (weekday)
  lbd2   = Lambda(lambda x: x[:,:, n_streets:n_streets+n_weeks])(dense1)
  out_2  = GumbelSoftmax(units=n_weeks,name='out_2') (lbd2)

  # third discret variable (slot)
  lbd3   = Lambda(lambda x: x[:,:,n_streets+n_weeks:n_streets+n_weeks+n_slots])(dense1)
  out_3  = GumbelSoftmax(units=n_slots,name='out_3') (lbd3)

  # continuos variable
  lbd4   = Lambda(lambda x: x[:,:,-1:])(dense1)
  out_4  = Dense(units=1,activation='relu',name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  permute = Permute((2,1), name="permute1")(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=permute)

  return model

def generator_model_v17(latent_dim):
  model = Sequential()
  n_nodes = 128 * 4 * 4
  model.add(Dense(n_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((4,4,128)))
  # upsample to 8x4
  model.add(Conv2DTranspose(128, (4,4), strides=(2,1), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # upsample to 16x4
  model.add(Conv2DTranspose(128, (4,4), strides=(2,1), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  # model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
  model.add(Conv2DTranspose(128, (4,4), strides=(3,1), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(1, (4,4), activation='sigmoid', padding='same'))
  return model