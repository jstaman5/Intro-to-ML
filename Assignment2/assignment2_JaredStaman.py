#Jared Staman
#CS 425 Assignment 2

import pandas as pd
import csv
import numpy as np

def main():
    
    with open('height_weight.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                #heights = row[0]
                #weights = row[1]
                sexes = convert_sex(row[2])
                line_count += 1
        
    


# convert Male to 1, Female to 0
def convert_sex(sex):
    if(sex == " Male"):
        sex = 1
    else:
        sex = 0
    
    return sex

main()