#Jared Staman
#CS 425 Assignment 4: Linear Regression

import math
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.set()

# read data from file, produce mpg related info and plots

data = pd.read_csv('auto-mpg.csv', sep=',')

data['Country_code']=data.origin.replace([1,2,3],['USA','Europe','Japan'])

mpg = data['mpg'].values
cylinders = data['cylinders'].values
displacement = data['displacement'].values
horsepower = data['horsepower'].values
weight = data['weight'].values
acceleration = data['acceleration'].values

#pd.describe()
sbn.heatmap() #-- use show data.corr() output
sbn.pairplot() #-- consider hue='Country_code'

# create data arrays used below, split into train/test, standardize and prepend by 1s
'''
X = np.array([ ... ], dtype=np.float32).T
y = np.array(...)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# model evaluation - MSE criterion function
def J(y, y_pred, w, beta):
  ...
  pass
  #return Jdata, beta*Jmodel, Jdata + beta*Jmodel

# model evaluation - R2 statistical score
def r2(y, y_pred):
  mean_y = np.mean(y)
  ss_res = np.sum(np.square(y-y_pred))
  ss_tot = np.sum(np.square(y-mean_y))
  return 1 - (ss_res / ss_tot)

# model training - MSE gradient descent
def gd(X, y, eta, beta, eps=0.01, kMax=100000):
  ...
  pass
  #return w, w_change, k

# evaluate hyperparameters on training data

eta_values =  [ 0.001, 0.01, 0.1 ]
beta_values = [ 0.0, 0.1, 0.5, 1.0 ]


for eta, beta combinations:
  ...

produce plots

# select eta and beta, retrain and run test data

eta = ...
beta = ...

plt.scatter()

'''