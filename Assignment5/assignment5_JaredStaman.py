#Jared Staman
#CS 425 ML: Bayesian Classification

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .20)

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
    print(male_covariance)
    print(female_covariance)
    
    return


if __name__ == "__main__":
    main()