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

def main():
    
    df = pd.read_csv("height_weight.csv")

    #data = dframe.to_numpy

    height = "Height (in)"
    weight = " Weight (lbs)"
    sex = " Sex"
    
    male = df[df[sex] == " Male"].to_numpy()
    female = df[df[sex] == " Female"].to_numpy()

    min_male_height = male[:,0].min()
    max_male_height = male[:,0].min()
    min_male_weight = male[:,1].max()
    mix_male_weight = male[:,1].max()

    min_female_height = female[:,0].min()
    max_female_height = female[:,0].min()
    min_female_weight = female[:,1].max()
    mix_female_weight = female[:,1].max()

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


    #weight [:,1]
    #print(min_male_height)
    
    #change colors?
    sns.scatterplot(data = df, x = weight, y = height, hue = sex)

    plt.show()

    
    #x,y = np.meshgrid(np.arange())



main()