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

    #male[:,2] = 1
    #female[:,2] = 0
    df[" Sex"].replace({" Male":0, " Female":1}, inplace=True)

    #print(df)                                     

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
    #sns.scatterplot(data = df, x = weight, y = height, hue = sex, palette = ['blue', 'red'] )

    #plt.show()

    #2d array of heights and weights
    X = df.iloc[:, 0:2].values
    
    #1d array male or female
    Y = df.iloc[:,2].values

    #split data 80, 20
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .20)
    
    #create model  (look into changing hyper-paramters)
    classifier = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')
    
    #Kfold to test hyper parameters
    K = 10
    train_res = np.zeros(K)
    test_res = np.zeros(K)
    best_avg = 0

    #genderless std of height and weight
    std_height = df.to_numpy()[:,0].std()
    std_weight = df.to_numpy()[:,1].std()
    
    param_df = pd.DataFrame()

    #KFold
    '''
    kfold = KFold(n_splits=K, shuffle=True)
    for neighbors in range (5, 35, 5):
      for weight in ['distance', 'uniform']:
        classifier.set_params(n_neighbors = neighbors, weights = weight)
        for k, (train_index, test_index) in enumerate(kfold.split(x_train)):

          x_train2, x_test2 = x_train[train_index], x_train[test_index]
          y_train2, y_test2 = y_train[train_index], y_train[test_index]

          x_train2[:,0] /= std_height
          x_train2[:,1] /= std_weight

          x_test2[:,0] /= std_height
          x_test2[:,1] /= std_weight

          classifier.fit(x_train2, y_train2)  

          y_pred_train = classifier.predict(x_train2)  
          y_pred_test = classifier.predict(x_test2)  

          C_train = metrics.accuracy_score(y_train2, y_pred_train)  
          C_test = metrics.accuracy_score(y_test2, y_pred_test)
          
          train_res[k] = C_train  
          test_res[k] = C_test
          
          #print('k={}:  train {:0.3f} test {:0.3f}'.format(k, C_train, C_test))

        #Finding the best parameters
        avg = test_res.mean()
        if avg > best_avg:
          best_avg = avg
          best_neighbor = neighbors
          best_weight = weight


        data = {'Neighbors':[neighbors], 'Accuracy':[avg], 'Weights':[weight]}
        temp_df = pd.DataFrame(data)
        param_df = param_df.append(temp_df)
    #print('best neighbor={}, best weight={}, best avg={:0.3f}'.format(best_neighbor, best_weight, best_avg))
    '''
    #sns.scatterplot(data = param_df, x='Neighbors', y='Accuracy', hue='Weights')
    #plt.show()

    #New Model for test data
    #new_classifier = KNeighborsClassifier(n_neighbors = best_neighbor, weights = best_weight)
    new_classifier = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
    
    x_train[:,0] /= std_height
    x_train[:,1] /= std_weight

    x_test[:,0] /= std_height
    x_test[:,1] /= std_weight
    new_classifier.fit(x_train, y_train)

    y_pred_train = new_classifier.predict(x_train)
    y_pred_test = new_classifier.predict(x_test)

    #Classificatin Accuracy
    C_train = metrics.accuracy_score(y_train, y_pred_train)
    C_test = metrics.accuracy_score(y_test, y_pred_test)
    '''
    print('train {:0.3f}  test {:0.3f}'.format(C_train, C_test))
    '''
    #Visualize Results
    xx, yy = np.meshgrid(np.arange(min_female_weight/std_weight, max_male_weight/std_weight, .1), np.arange(min_female_height/std_height, max_male_height/std_height, .1))

    Z = new_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='bwr', alpha = .3)
    plt.title("Male and Female Decision Regions")
    plt.plot()

    plot_df = pd.DataFrame(x_test, columns = ['Height', 'Weight'])
    sexes = ["Male" if x==0 else "Female" for x in y_test]
    plot_df['Sex'] = sexes
    sns.scatterplot(data=plot_df, x='Weight', y='Height', hue='Sex', palette = ['red', 'blue'] )
    plt.show()
    
main()