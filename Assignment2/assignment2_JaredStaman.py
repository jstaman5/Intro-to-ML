#Jared Staman
#CS 425 Assignment 2

import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics

def main():
    
    df = pd.read_csv("height_weight.csv")

    

    height = "Height (in)"
    weight = " Weight (lbs)"
    sex = " Sex"
    
    male = df[df[sex] == " Male"].to_numpy()
    female = df[df[sex] == " Female"].to_numpy()

    #print(male)

    min_male_height = male[:,0].min()
    max_male_height = male[:,0].max()
    min_male_weight = male[:,1].min()
    max_male_weight = male[:,1].max()

    min_female_height = female[:,0].min()
    max_female_height = female[:,0].max()
    min_female_weight = female[:,1].min()
    max_female_weight = female[:,1].max()

    male_height_mean = male[:,0].mean()
    male_weight_mean = male[:,1].mean()
    female_height_mean = female[:,0].mean()
    female_weight_mean = female[:,1].mean()

    male_height_std = male[:,0].std()
    male_weight_std = male[:,1].std()
    female_height_std = female[:,0].std()
    female_weight_std = female[:,1].std()

    male_height_snr = male_height_mean / male_height_std
    male_weight_snr = male_weight_mean / male_weight_std
    female_height_snr = female_height_mean / female_height_std
    female_weight_snr = female_weight_mean / female_weight_std

    
    #add title?
    sns.scatterplot(data = df, x = weight, y = height, hue = sex, palette = ['blue', 'red'] )

    #plt.show()

    #2d array of heights and weights
    X = df.iloc[:, 0:2].values
    
    #1d array male or female
    Y = df.iloc[:,2].values

    #split data 80, 20
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .20)
    
    #create model  (look into changing hyper-paramters)
    classifier = KNeighborsClassifier(n_neighbors = 30, weights = 'uniform')
    
    #fit model
    classifier.fit(x_train, y_train)

    y_pred_train = classifier.predict(x_train)
    y_pred_test = classifier.predict(x_test)

    #Classification Accuracy
    C_train = metrics.accuracy_score(y_train, y_pred_train)
    C_test = metrics.accuracy_score(y_test, y_pred_test)

    print('train {:0.3f}  test {:0.3f}'.format(C_train, C_test))
    
    #Visualize Results
    xx, yy = np.meshgrid(np.arange(min_female_weight, max_male_weight, step = .02), 
                        np.arange(min_female_height, max_male_height, step = .02))

    from matplotlib.colors import ListedColormap


   # plt.contourf(xx, yy, Z)

  #  plt.show()
main()