#Jared Staman
#CS 425 Assignment 4: Linear Regression

import math
import numpy as np
from numpy.lib.function_base import gradient 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn


# model evaluation - MSE criterion function
def J(y, y_pred, w, beta):
  #X * w = y_pred
  T = np.transpose(y_pred - y)
  Jdata = (1 / (2 * np.size(y))) * np.dot(T, y_pred - y)

  T = np.transpose(w)
  Jmodel = np.dot(T, w)

  return Jdata, beta*Jmodel, Jdata + beta*Jmodel

# model evaluation - R2 statistical score
def r2(y, y_pred):
  mean_y = np.mean(y)
  ss_res = np.sum(np.square(y-y_pred))
  ss_tot = np.sum(np.square(y-mean_y))
  return 1 - (ss_res / ss_tot)

# model training - MSE gradient descent
def gd(X, y, eta, beta, eps=0.01, kMax=100000):
  print_check = False
  w_max_change = []
  N, M = np.shape(X)
  w = np.zeros(M)
  prev_w = np.zeros(M)
  
  for k in range(kMax):
    
    prev_w = w
    T = np.transpose(X)
    pred = np.dot(X, w)
    
    gradient = ((1/N) * np.dot(T , pred - y)) + ( 2 * beta * w)
    
    w = prev_w - (eta * gradient)
  
    w_max_change.append(max(abs(prev_w - w)))
    #run until max change in any weight vector variable is less than epsilon
    if w_max_change[k] < eps:
      break
    
    if print_check == True:
      Jdata, beta_Jmodel, sum_of_Jdata_model = J(y, pred, w, beta)
      print(Jdata, beta_Jmodel, sum_of_Jdata_model, w_max_change[k])

  return w, w_max_change, k


#print(gd(X_train, y_train, .001, .01))


# evaluate hyperparameters on training data


def main():

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

  #info = data.describe()
  #print(info)
  #sbn.heatmap(data.corr()) #-- use show data.corr() output
  #plt.show()
  #sbn.pairplot(data, hue = 'Country_code') #-- consider hue='Country_code'
  #plt.show()


  # create data arrays used below, split into train/test, standardize and prepend by 1s

  def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

  cylinders = normalize(cylinders)
  displacement = normalize(displacement)
  horsepower = normalize(horsepower)
  weight = normalize(weight)
  acceleration = normalize(acceleration)
  #mpg = normalize(mpg)
  X = np.array([ np.ones(shape = cylinders.shape), cylinders, displacement, horsepower, weight, acceleration ], dtype=np.float32).T
  y = np.array(mpg)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


  eta_values =  [ 0.001, 0.01, 0.1 ]
  beta_values = [ 0.0, 0.01, 0.05, 0.1, 0.5, 1.0 ]
  eps_values =  [ .001, .01 ]

  best_eta = 0
  best_beta = 0
  best_r2 = 0

  # 6. find best hyperparameters for eps = .01
  for eta in eta_values:
    iterations = []
    fidelity = []
    model_term = []
    summation = []
    for beta in beta_values:
      w, w_max_change, k = gd(X_train, y_train, eta, beta)
      y_pred = np.dot(X_train, w)
      r2_score = r2(y_train, y_pred)

      iterations.append(k)
      j, jd, s = J(y_train, y_pred, w, beta)
      fidelity.append(j)
      model_term.append(jd)
      summation.append(s)

      if r2_score > best_r2:
        best_r2 = r2_score
        best_eta = eta
        best_beta = beta


    fig, ax = plt.subplots(1, 2)
    #plot iterations vs beta
    ax[0].scatter(beta_values, iterations)
    ax[0].set_title('iterations vs beta')
    ax[0].set_xlabel("Beta values")
    ax[0].set_ylabel("Iterations")
    #plot MSE minimization results vs beta
    ax[1].set_title('MSE minimization results')
    width = .25
    eps = .01 #loop through
    fig.suptitle("Eta = {}, Eps = {}".format(eta, eps))
    labels = ['0.0', '0.0', '0.01', '0.05', '0.1', '0.5', '1.0' ]
    x = np.arange(len(beta_values))
    ax[1].bar(x-width, fidelity, width, label = 'Fidelity')
    ax[1].bar(x, model_term, width, label = 'Model Term')
    ax[1].bar(x + width, summation, width, label = 'Sum')
    ax[1].legend()
    ax[1].set_xticklabels(labels)


    w, w_max_change, k = gd(X_train, y_train, best_eta, best_beta, .01)
    y_pred = np.dot(X_test, w)
    plt.figure()
    plt.scatter(np.arange(1, np.size(y_test)+1), y_pred, label = "")
  
  plt.show()
  print(best_eta, best_beta, iterations, best_r2)

'''
for eta, beta combinations:
  ...

produce plots

# select eta and beta, retrain and run test data

eta = ...
beta = ...

plt.scatter()

'''

if __name__ == "__main__":
  main()