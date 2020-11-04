# from keras.layers import R
from keras.layers import Input, LSTM, Lambda, Dense, Dropout
from keras.models import Model
from keras.initializers import RandomNormal
from keras.initializers import RandomUniform
from keras.constraints import Constraint
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Sequential
from generators import generate_latent_points, get_slot_range
from utils import GumbelSoftmax, hstack_samples
import numpy as np



def generate_fake_samples_wasserstain(g_model,n_samples,n_steps=None,n_features=None):
  # generate points in the latente space
  x_input = generate_latent_points(n_samples,n_steps,n_features)
  # predict outputs
  X = g_model.predict(x_input)
  # create 'fake' class labels (1)
  y = np.ones((n_samples,1))
  return X,y

def generate_real_samples_wasserstain(n_samples,dataset):
  start = np.random.randint(0,len(dataset)-n_samples,size=1)[0]
  X = dataset[start:start+n_samples]

  # labels for the real samples (-1)
  y = -np.ones((n_samples,1))
  return X,y

def wasserstain_loss(y_true,y_pred):
  return K.mean(y_true*y_pred)

class ClipConstraint(Constraint):
  def __init__(self,clip_value):
    self.clip_value = clip_value
  
  def __call__(self,weights):
    return K.clip(weights,-self.clip_value, self.clip_value)
  
  def get_config(self):
    return {'clip_value':self.clip_value}

def define_generator_wasserstain(n_streets=2, n_weeks=7, interval=30, n_features=None):
  
  """
  Defines the generator model for the wasserstain GAN
  ------
  params: 
  - n_streets (number of streets)
  - n_weeks (number of weeks)
  - interval (time interval that builds the dataset) 
  - n_features (number of features)
  -----
  returns: the NN model for the generator
  """
  #init = RandomNormal(stddev=0.02)
  init = RandomUniform(minval=0,maxval=1)
  n_slots = len(get_slot_range(interval))
  n_steps = n_streets+n_weeks+n_slots+1
  
  # inverted because is a multi-input multi-step
  visible = Input(shape=(n_features,n_steps))

  lstm = LSTM(n_steps, return_sequences=True,kernel_initializer=init,name='lstm')(visible)

  # first discrete variable (street)
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
  out_4  = Dense(units=1,activation='relu',kernel_initializer=init, name='out_4') (lbd4)

  merge = concatenate([out_1, out_2, out_3, out_4])

  hstack = Lambda(hstack_samples)(merge)
  
  # size of all discrete variable + the continuos variable
  #output  = Dense(n_features)(merge)
  model = Model(inputs=visible, outputs=hstack)

  return model

def define_critic(n_hiddens,n_steps=None,n_features=None):
  # init = RandomNormal(stddev=0.02)
  init = RandomUniform(minval=0,maxval=1)
  const = ClipConstraint(0.01)
  visible = Input(shape=(n_steps,n_features))
  lstm_1 = LSTM(n_hiddens,activation='relu',kernel_initializer=init,kernel_constraint=const,name='lstm_1') (visible)
  drop_1 = Dropout(0.2) (lstm_1)
  # maybe add dropout
  # lstm_2 = LSTM(20,activation='relu', kernel_initializer=init,kernel_constraint=const, name='lstm_2') (drop_1)
  # maybe add dropout
  out   = Dense(1) (drop_1)
  model = Model(inputs=visible,outputs=out)
  #opt = Adam(lr=0.0001,beta_1=0.5)
  #opt = Adam()
  opt = RMSprop(lr=0.00005)
  # binary_crossentropy, to classify fake (0) and real (1) samples
  model.compile(optimizer=opt, loss=wasserstain_loss)
  return model

def define_wasserstain_gan(g_model,c_model):
  c_model.trainable = False
  model = Sequential()
  model.add(g_model)
  model.add(c_model)
  opt = RMSprop(lr=0.00005)
  model.compile(loss=wasserstain_loss, optimizer=opt)
  return model



def train_wasserstain_gan(g_model,c_model, gan_model, dataset, 
                          n_epochs=10, n_batch=64,n_critic=5,n_steps=None,n_features=None):
  bat_per_epo = int(dataset.shape[0]/n_batch)
  n_steps_training = bat_per_epo*n_epochs
  half_batch = int(n_batch/2)
  c1_hist, c2_hist,g_hist = list(),list(),list()
  for i in range(n_steps_training):
    c1_tmp,c2_tmp = list(),list()
    for _ in range(n_critic):
      X_real,y_real = generate_real_samples_wasserstain(half_batch, dataset)
      c_loss1 = c_model.train_on_batch(X_real,y_real)
      c1_tmp.append(c_loss1)
      X_fake,y_fake = generate_fake_samples_wasserstain(g_model,half_batch,n_steps,n_features)
      c_loss2 = c_model.train_on_batch(X_fake,y_fake)
      c2_tmp.append(c_loss2)
    c1_hist.append(np.mean(c1_tmp))
    c2_hist.append(np.mean(c2_tmp))

    X_gan = generate_latent_points(n_batch,n_steps,n_features)
    y_gan = -ones((n_batch,1))
    g_loss= gan_model.train_on_batch(X_gan, y_gan)
    g_hist.append(g_loss)
    print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))

    if ((i>0) and (i%20)==0):
      plot_training_wgan(i,loss_c1=c1_hist,loss_c2=c2_hist,loss_g=g_hist)