from pathlib import Path

import json
import matplotlib as plt
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from tensorflow_probability import distributions
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
"""## Gumbel-softmax Layer"""

class GumbelSoftmax(Layer):
  """
  Gumbel-softmax layer
  """
  def __init__(self,units=32,**kwargs):
    super(GumbelSoftmax, self).__init__(**kwargs)
    self.temperature = 1e-5 # makes the distribution discrete

  def gumbel_softmax(self,logits):
    """
    Returns a gumbel-softmax distribution for the logits provided
    ----
     -params: logits (array of real numbers)
     -returns: a sample of the gumbel-softmax distribution
    """
    dist = distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)
    return dist.sample()    

  def call(self, inputs):
    dist = self.gumbel_softmax(inputs)
    return dist

def get_slot_range(interval=30):
  """
  Get the slots for the interval provided
  """
  minutes_day = 24*60
  slots = np.arange(0,minutes_day+1,interval)
  slots_range = range(0,len(slots))
  return slots_range

"""## Default GAN"""
def plot_training(epoch,range_training=[], loss_d=[],
                  loss_g=[],labels=['G','D'],path=None,today=None, image_title="",n_teste=None):
  """
  Plot training 
  """
  fig = plt.figure()
  plt.title('e-{} '.format(epoch)+image_title)
  plt.plot(loss_g,label=labels[0],color='red')
  plt.plot(loss_d,label=labels[1],color='green')
  plt.xlabel('Batches')
  plt.ylabel('Loss')
  plt.legend()
  if path:
    path_to_save = path+"/{}_{}_{}_t{}".format(today.day, today.month,
                                                today.year,n_teste)
    Path(path_to_save).mkdir(parents=True,exist_ok=True)
    plt.savefig(path_to_save+"/{}".format(epoch))
  plt.show()
  plt.close(fig)

def save_training(base_url,data,filename):
  now = datetime.now()
  today = "/{}_{}_{}".format(now.day,now.month,now.year)
  path_to_save = base_url+today
  Path(path_to_save).mkdir(parents=True,exist_ok=True)
  with open(path_to_save+"/"+filename,mode='w') as fp:
    json.dump(data,fp)


def convert_data(dataset):
  ruas = ['Av. Mal. Campos', 'Av. Nossa Senhora da Penha']
  wk = [0,1,2,3,4,5,6]
  slots = [x for x in range(49)]

  ruas = np.array(ruas).reshape(-1,1)
  wk = np.array(wk).reshape(-1,1)
  slots = np.array(slots).reshape(-1,1)

  encoderStreets = OneHotEncoder(sparse=False)
  encoderStreets = encoderStreets.fit(ruas)

  encoderWeekDay = OneHotEncoder(sparse=False)
  encoderWeekDay = encoderWeekDay.fit(wk)

  encoderSlots   = OneHotEncoder(sparse=False)
  encoderSlots   = encoderSlots.fit(slots)


  convertido = []
  for arr in dataset:
    data0 = np.concatenate((arr.reshape(arr.shape[0],arr.shape[1],1)),axis=1)
    interno = []
    for a in data0:
      a1 = encoderStreets.inverse_transform(a[:2].reshape(-1,2))
      a2 = encoderWeekDay.inverse_transform(a[2:9].reshape(-1,7))
      a3 = encoderSlots.inverse_transform(a[9:-1].reshape(-1,49))
      a4 = [[a[-1]]]
      resultado = np.concatenate((a1,a2,a3,a4),axis=1).reshape(4,)
      interno.append(resultado)
    convertido.append(interno)
  convertido = np.array(convertido)
  return convertido


def generate_latent_points(n_samples, n_steps, n_features):
  # generate points in the latent space
  x_input = np.random.randn(n_samples,n_steps,n_features)
  return x_input
  
def generate_fake_samples(g_model=None, n_samples=None, n_steps=None, n_features=None):
  # generate points in the latente space
  x_input = generate_latent_points(n_samples,n_steps,n_features)
  # predict outputs
  X = g_model.predict(x_input)
  # create 'fake' class labels (0)
  y = np.zeros((n_samples,1))
  return X, y