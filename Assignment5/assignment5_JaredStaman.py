#Jared Staman
#CS 425 ML: Bayesian Classification

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math

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
    male_covariance = np.cov(male_x_train)
    female_covariance = np.cov(female_x_train)
    pooled_covariance = np.cov(x_train)
    
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

    def F(x, mu, sigma):
        vector_erf = np.vectorize(math.erf)
        return .5 * (1+vector_erf (((x - mu)/ sigma)/ math.sqrt(2)))

    #thresholds
    threshold_height = qsolve(male_height_mean, male_height_std, male_probability, female_height_mean, female_height_std, female_probability)
    threshold_weight = qsolve(male_weight_mean, male_weight_std, male_probability, female_weight_mean, female_weight_std, female_probability)

    #calculate probability errors
    pE_height = (1-F(x_train, male_height_mean, male_height_std)) * male_probability + F(x_train, female_height_mean, female_height_std) * female_probability
    
    pE_weight = (1-F(x_train, male_weight_mean, male_weight_std)) * male_probability + F(x_train, female_weight_mean, female_weight_std) * female_probability

    #Classify Test Data
    #pred_weight_male = x_test[np.where(x_test[:,1] > threshold_weight)]
    #pred_height_male = x_test[np.where(x_test[:,0] > threshold_height)]
    pred_weight = np.array([1 if weight > threshold_weight else 0 for weight in x_test[:,1]])
    pred_height = np.array([1 if height > threshold_height else 0 for height in x_test[:,0]])
    #pred_weight_female = x_test[np.where(x_test[:,1] <= threshold_weight)]
    #pred_height_female = x_test[np.where(x_test[:,0] <= threshold_height)]
    
    #Create confusion matrix
    h_tn, h_fp, h_fn, h_tp = confusion_matrix(y_test, pred_height).ravel()
    w_tn, w_fp, w_fn, w_tp = confusion_matrix(y_test, pred_weight).ravel()

    print(h_tn, h_fp, h_fn, h_tp)
    #Classification Accuracy and Error Rates
    return


if __name__ == "__main__":
    main()