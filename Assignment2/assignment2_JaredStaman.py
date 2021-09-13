#Jared Staman
#CS 425 Assignment 2

import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    
    df = pd.read_csv("height_weight.csv")

    #data = dframe.to_numpy

    height = "Height (in)"
    weight = " Weight (lbs)"
    sex = " Sex"
    
    male = df[df[sex] == " Male"].to_numpy()
    female = df[df[sex] == " Female"].to_numpy()

    min_male_height = male[:,0].min()
    #weight [:,1]
    #print(min_male_height)
    
    sns.scatterplot(data = df, x = weight, y = height, hue = sex)

    plt.show()
main()