#Jared Staman
#CS 425 ML: Bayesian Classification

from os import error
from numpy.lib.function_base import cov
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
from numpy.linalg import det, inv
import matplotlib.pyplot as plt

def main():

    #read in data
    df = pd.read_csv("height_weight.csv")

    height = "Height (in)"
    weight = " Weight (lbs)"
    sex = " Sex"

    #put data into numpy arrays
    male = df[df[sex] == " Male"].to_numpy()
    female = df[df[sex] == " Female"].to_numpy()
    df[" Sex"].replace({" Male":0, " Female":1}, inplace=True)

    #get height and weights separated from gender
    X = df.iloc[:, 0:2].values
    y = df.iloc[:, 2].values

    #split test and training data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 425)

    male_x_train = x_train[np.where(y_train == 0)]
    female_x_train = x_train[np.where(y_train == 1)]

    #separate training data into male and female
    '''
    for i in range(y_train.size):
        if(y_train[i] == 0):
            male_x_train.append(x_train[i])
        else:
            female_x_train.append(x_train[i])

    #convert to numpy arrays from lists        
    male_x_train = np.asarray(male_x_train)
    female_x_train = np.asarray(female_x_train)
    '''

    #calculate means
    male_height_mean, male_weight_mean = np.mean(male_x_train, axis = 0)
    female_height_mean, female_weight_mean = np.mean(female_x_train, axis = 0)
    pooled_height_mean, pooled_weight_mean = np.mean(x_train, axis = 0)
    
    #calculate covariance matrices
    male_covariance = np.cov(male_x_train, rowvar = False)
    female_covariance = np.cov(female_x_train, rowvar = False)
    pooled_covariance = np.cov(x_train, rowvar = False)
    
    #determine prior class probabilities from training data
    male_probability = np.size(male_x_train) / np.size(x_train)
    female_probability = np.size(female_x_train) / np.size(x_train)

    #calculate standard deviation
    male_height_std, male_weight_std = np.std(male_x_train, axis = 0)
    female_height_std, female_weight_std = np.std(female_x_train, axis = 0)
    pooled_height_std, pooled_weight_std = np.std(x_train, axis = 0)

    #solve quadratic equatin needed to determine 1D Bayesian Classification Threshold
    def qsolve(mu1, sigma1, P1, mu2, sigma2, P2):
        A = -1/2 * (1/sigma1**2 - 1/sigma2**2)
        B = (mu1/sigma1**2 - mu2/sigma2**2)
        C = -1/2 * ((mu1/sigma1)**2 - (mu2/sigma2)**2) - math.log(sigma1/sigma2) + math.log(P1/P2)

        return (-B + math.sqrt(B**2 - 4*A*C)) / (2*A)

    #Used for graphing the gaussian curves
    def p(x, mu, sigma):
        vector_exp = np.vectorize(math.exp)
        z = 1/(math.sqrt(2*np.pi)*sigma)
        return z*vector_exp(-0.5*((x-mu)/sigma)**2)

    def F(x, mu, sigma):
        vector_erf = np.vectorize(math.erf)
        return .5 * (1+vector_erf (((x - mu)/ sigma)/ math.sqrt(2)))

    #thresholds
    threshold_height = qsolve(male_height_mean, male_height_std, male_probability, female_height_mean, female_height_std, female_probability)
    threshold_weight = qsolve(male_weight_mean, male_weight_std, male_probability, female_weight_mean, female_weight_std, female_probability)

    #calculate probability errors
    pE_height = (1-F(x_train, male_height_mean, male_height_std)) * male_probability + F(x_train, female_height_mean, female_height_std) * female_probability
    #pE_height2 = (1-F(threshold_height, male_height_mean, male_height_std)) * male_probability + F(threshold_height, female_height_mean, female_height_std) * female_probability
    pE_weight = (1-F(x_train, male_weight_mean, male_weight_std)) * male_probability + F(x_train, female_weight_mean, female_weight_std) * female_probability

    #Classify Test Data
    pred_weight = np.array([1 if weight > threshold_weight else 0 for weight in x_test[:,1]])
    pred_height = np.array([1 if height > threshold_height else 0 for height in x_test[:,0]])
    
    
    #Create confusion matrix
    h_tn, h_fp, h_fn, h_tp = confusion_matrix(y_test, pred_height).ravel()
    w_tn, w_fp, w_fn, w_tp = confusion_matrix(y_test, pred_weight).ravel()

    #Classification Accuracy and Error Rates
    height_acc = h_tp + h_tn / np.size(y_test)
    weight_acc = w_tp + w_tn / np.size(y_test)

    height_err = h_fp + h_fn / np.size(y_test)
    weight_err = w_fp + w_fn / np.size(y_test)

    #plots
    
    '''hist = plt.histogram(x_train[:,0], bins = 10) #, range = (np.min(x_train[:,0], np.max(x_train[:,0]))))
    plt.plot(hist)
    plt.show()'''

    #height graph with histogram on top
    '''male_height_range = np.array([x for x in range(50, 80)])
    female_height_range = np.array([x for x in range(50,80)])
    p1 = p(male_height_range, male_height_mean, male_height_std) * male_probability
    p2 = p(female_height_range, female_height_mean, female_height_std) * female_probability
    plt.plot(male_height_range, p1, '-b')
    plt.plot(female_height_range, p2, '-r')
    plt.axvline(x = threshold_height, linestyle = 'dashed')
    plt.show()
    plt.figure()'''
    '''
    #weight graph with histogram on top
    male_weight_range = np.array([x for x in range(60, 250)])
    female_weight_range = np.array([x for x in range(60,250)])
    p1 = p(male_weight_range, male_weight_mean, male_weight_std) * male_probability
    p2 = p(female_weight_range, female_weight_mean, female_weight_std) * female_probability
    plt.plot(male_weight_range, p1, '-b')
    plt.plot(female_weight_range, p2, '-r')
    plt.axvline(x = threshold_weight, linestyle = 'dashed')
    plt.show()'''

    #linear 2D Bayesian classifier
    def linear(mu, mu2, covariance, prior):
        inverse = inv(covariance)
        w0 = -1/2 * (np.dot(np.dot(np.transpose(mu + mu2), inverse) , (mu-mu2))) + math.log(prior)
        w = np.dot(inverse, mu-mu2)
        
        return np.transpose(w), w0

    def quadratic(mu, mu2, covariance, prior):
        inverse = inv(covariance)
        w0 = -1/2 * np.dot(np.dot(np.transpose(mu+mu2), inverse) , mu-mu2) - ((1/2) * math.log(det(covariance))) + math.log(prior)
        W = -1/2 * (inverse)
        w = np.dot(inverse, mu-mu2)
    
        return W.flatten(), w, w0

    pooled_covariance[0][1] = 0
    pooled_covariance[1][0] = 0
    
    male_mu = np.transpose(np.array([[male_height_mean, male_weight_mean]]))
    female_mu = np.transpose(np.array([[female_height_mean, female_weight_mean]]))
    
    w, w0 = linear(male_mu, female_mu, pooled_covariance, male_probability/female_probability)
    slope = w[0][0] / w[0][1] * -1
    #print(slope)
    b = w0[0][0] / w[0][1] * -1
    #print(b)

    W, w, w0 = quadratic(male_mu, female_mu, pooled_covariance, male_probability/female_probability)
    print(W, w, w0)
  
    '''pred = np.array([1 if male_linear[i] >= female_linear[i] else 0 for i in range(len(x_train))])
    fp, tp, tn, fn = confusion_matrix(y_train, pred).ravel()
    accuracy = (tp + tn) / np.size(pred)
    err = (fp + fn) / np.size(pred)'''
    
    return


if __name__ == "__main__":
    main()