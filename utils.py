from pathlib import Path

import json
import matplotlib.pyplot as plt
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
  # TODO: Remove units
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
def plot_training(epoch, loss_d=[], loss_g=[],labels=['G','D'], path=None,today=None, image_title="",n_teste=None):
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
  # plt.show()
  # plt.close(fig)

def save_training(base_url,data,filename):
  now = datetime.now()
  today = "/{}_{}_{}".format(now.day,now.month,now.year)
  path_to_save = base_url+today
  Path(path_to_save).mkdir(parents=True,exist_ok=True)
  with open(path_to_save+"/"+filename,mode='w') as fp:
    json.dump(data,fp)


def convert_data(dataset,encoderStreets,encoderWeekDay,encoderSlots):
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

def generate_latent_points3(n_samples, latent_dim):
  # generate points in the latent space
  x_input = np.random.randn(n_samples*latent_dim)
  x_input = x_input.reshape(n_samples,latent_dim)
  return x_input

def generate_fake_samples3(g_model, n_samples, latent_dim):
  x_input = generate_latent_points3(n_samples, latent_dim)
  X = g_model.predict(x_input)
  y = np.zeros((n_samples,1))
  return X, y

def get_real_samples(n_samples,dataset):
  """
  Returns n_samples real samples for the dataset
  ----
  params: 
  - n_samples
  - dataset
  """

  start = np.random.randint(0,len(dataset)-n_samples,size=1)[0]
  X = dataset[start:start+n_samples]
  y = np.ones((n_samples,1))
  return X,y

def get_real_samples3(n_samples, dataset, wkday=0):
  real_samples = []
  full = False
  while not full:
    r,_ = get_real_samples(n_samples*10,dataset)
    for i, arr in enumerate(r):
      # compares the weekday
      equals = arr[:,1][0] == wkday
      if equals:
        real_samples.append(arr)
        if (len(real_samples) == n_samples): 
          full = True
          break
  real_samples = np.array(real_samples)
  real_samples = real_samples.reshape(n_samples,48,4,1)
  y = np.ones((n_samples,1))
  return real_samples, y


def plot_img(d,base_url=None,testname="",ep=0):
  today = datetime.now()
  path = base_url+"/{}_{}_{}/{}".format(today.day, today.month, today.year,testname)
  
  Path(path).mkdir(parents=True,exist_ok=True)
  fig = plt.figure()  
  plt.imshow(d,aspect="auto")
  plt.show()  
  plt.savefig(path+"/{}.png".format(ep))
  plt.close(fig)