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
    male_x_train = x_train[np.where(y_train == 0)]
    female_x_train = x_train[np.where(y_train == 1)]
    

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
    h_fp, h_tp, h_tn, h_fn = confusion_matrix(y_test, pred_height).ravel()
    w_fp, w_tp, w_tn, w_fn = confusion_matrix(y_test, pred_weight).ravel()
    #print(confusion_matrix(y_test, pred_weight))

    #Classification Accuracy and Error Rates
    height_acc = (h_tp + h_tn) / np.size(y_test)
    weight_acc = (w_tp + w_tn) / np.size(y_test)
    height_err = (h_fp + h_fn) / np.size(y_test)
    weight_err = (w_fp + w_fn) / np.size(y_test)
    '''print("1D height accuracy" ,height_acc)
    print("1D height error", height_err)
    print("1D weight accuracy" ,weight_acc)
    print("1D weight error", weight_err)'''
    



    # ********plots************

    #height graph with histogram on top
    male_height_range = np.array([x for x in range(50, 80)])
    female_height_range = np.array([x for x in range(50,80)])
    p1 = p(male_height_range, male_height_mean, male_height_std) * male_probability
    p2 = p(female_height_range, female_height_mean, female_height_std) * female_probability
    plt.plot(male_height_range, p1, '-b')
    plt.plot(female_height_range, p2, '-r')
    plt.title("P(Height|Sex) P(Sex)")
    plt.xlabel("Height (in)")
    plt.ylabel("Probability")
    plt.axvline(x = threshold_height, linestyle = 'dashed', color = "black")
    #plt.show()

    #histogram
    male_height_min = int(np.min(male_x_train[:,0])) - 2
    male_height_max = int(np.max(male_x_train[:,0])) + 2

    female_height_min = int(np.min(female_x_train[:,0])) - 2
    female_height_max = int(np.max(female_x_train[:,0])) + 2

    m_dist_range = np.arange(male_height_min, male_height_max, 0.5)
    m_probability = [(np.size(np.where((male_x_train[:,0]) >= x))- np.size(np.where((male_x_train[:,0]) > x + .5)))  \
    / len(male_x_train) for x in m_dist_range]

    f_dist_range = np.arange(female_height_min, female_height_max, 0.5)
    f_probability = [(np.size(np.where((female_x_train[:,0]) >= x))- np.size(np.where((female_x_train[:,0]) > x + .5)))  \
    / len(female_x_train) for x in f_dist_range]
    #print(probability)
    
    plt.step(m_dist_range, m_probability)
    plt.step(f_dist_range, f_probability)
    plt.show()

    plt.figure()
    
    #weight graph with histogram on top
    male_weight_range = np.array([x for x in range(60, 250)])
    female_weight_range = np.array([x for x in range(60,250)])
    p1 = p(male_weight_range, male_weight_mean, male_weight_std) * male_probability
    p2 = p(female_weight_range, female_weight_mean, female_weight_std) * female_probability
    plt.plot(male_weight_range, p1*10, '-b')
    plt.plot(female_weight_range, p2*10, '-r')
    plt.title("P(Weight|Sex) P(Sex)")
    plt.xlabel("Weight (lb)")
    plt.ylabel("Probability")
    plt.axvline(x = threshold_weight, linestyle = 'dashed', color = "black")

    male_weight_min = int(np.min(male_x_train[:,1])) - 2
    male_weight_max = int(np.max(male_x_train[:,1])) + 2

    female_weight_min = int(np.min(female_x_train[:,1])) - 2
    female_weight_max = int(np.max(female_x_train[:,1])) + 2

    m_dist_range = np.arange(male_weight_min, male_weight_max, 5)
    m_probability = [(np.size(np.where((male_x_train[:,1]) >= x))- np.size(np.where((male_x_train[:,1]) > x + 5)))  \
    / len(male_x_train) for x in m_dist_range]

    f_dist_range = np.arange(female_weight_min, female_weight_max, 5)
    f_probability = [(np.size(np.where((female_x_train[:,1]) >= x))- np.size(np.where((female_x_train[:,1]) > x + 5)))  \
    / len(female_x_train) for x in f_dist_range]

    plt.step(m_dist_range, m_probability)
    plt.step(f_dist_range, f_probability)
    plt.show()

    #linear 2D Bayesian classifier
    def linear(mu, mu2, covariance, prior):
        inverse = inv(covariance)
        w0 = -1/2 * (np.dot(np.dot(np.transpose(mu + mu2), inverse) , (mu-mu2))) + math.log(prior)
        w = np.dot(inverse, mu-mu2)
        
        return np.transpose(w), w0

    #Quadratic 2D Bayesian classifier
    def quadratic(x_test, male_mu, female_mu, m_covariance, f_covariance, male_prior, female_prior):
        '''inverse = inv(covariance)
        #print(math.log(det(covariance)))
        w0 = -1/2 * np.dot(np.dot(np.transpose(mu+mu2), inverse) , mu-mu2) - ((1/2) * math.log(det(covariance))) + math.log(prior)
        W = -1/2 * (inverse)
        w = np.dot(inverse, mu-mu2)'''
        m_inverse = inv(m_covariance)
        f_inverse = inv(f_covariance)
        row, col = x_test.shape
        
        output = np.empty((row, 1))
        for i, x in enumerate(x_test):
            male_score = -1/2 * (np.dot(np.dot(x, m_inverse), np.transpose(x))) + np.dot(np.dot(np.transpose(male_mu), m_inverse), np.transpose(x)) - (1/2) * np.dot(np.dot(np.transpose(male_mu), m_inverse) , male_mu) - (1/2) * math.log(det(m_covariance)) + math.log(male_prior)
            female_score = -1/2 * (np.dot(np.dot(x, f_inverse), np.transpose(x))) + np.dot(np.dot(np.transpose(female_mu), f_inverse), np.transpose(x)) - (1/2) * np.dot(np.dot(np.transpose(female_mu), f_inverse) , female_mu) - (1/2) * math.log(det(f_covariance)) + math.log(female_prior)
            if( male_score > female_score):
                output[i][0] = 1
            else:
                output[i][0] = 0
        return output

    #force zeros
    pooled_covariance[0][1] = 0
    pooled_covariance[1][0] = 0

    #transpose means for easier equation
    male_mu = np.transpose(np.array([[male_height_mean, male_weight_mean]]))
    female_mu = np.transpose(np.array([[female_height_mean, female_weight_mean]]))
    
    #y = mx + b
    w, w0 = linear(male_mu, female_mu, pooled_covariance, male_probability/female_probability)
    #print(w, w0)
    m = w[0][0] / w[0][1] * -1
    #print(m)
    b = w0[0][0] / w[0][1] * -1
    #print(b)

    l_prediction = np.array([1 if y > (x*m + b) else 0 for x, y in x_test])
    cm = confusion_matrix(y_test, l_prediction)
    fp, tp, tn, fn = cm.ravel()
    l_accuracy = (tp + tn) / np.size(l_prediction)
    l_error = (fp + fn) / np.size(l_prediction)
    '''print(cm)
    print("Linear accuracy", l_accuracy)
    print("Linear error", l_error)'''


    
    q_prediction = quadratic(x_test, male_mu, female_mu, male_covariance, female_covariance, male_probability, female_probability)
    '''print("W" , W)
    print("w" , w)
    print("w0" , w0)'''

    #q_prediction = np.array([1 if y > (np.dot(np.dot(np.array([x,y]), W), np.array([x, y]).T) + np.dot(w.T, np.array([x,y]).T) + w0) else 0 for x, y in x_test])
    cm2 = confusion_matrix(y_test, q_prediction)
    fp, tp, tn, fn = cm2.ravel()
    q_accuracy = (tp + tn) / np.size(q_prediction)
    q_error = (fp + fn) / np.size(q_prediction)
    print(cm2)
    print("Quadratic accuracy", q_accuracy)
    print("Quadratic error", q_error)
    

    # ***********plots****************

    #Linear Classifier Plot
    xg = np.linspace(50, 80, 1000)
    yg = np.linspace(60, 250, 1000)
    xg, yg = np.meshgrid(xg, yg)
    Z = -yg + m*xg + b 
    
    fig0, ax0 = plt.subplots()
    ax0.contour(xg, yg, Z, [0])
    ax0.contourf(xg, yg, Z, levels = 1, cmap = 'bwr')
    ax0.set_title("Linear Classifier")
    ax0.set_xlabel("Height (in)")
    ax0.set_ylabel("Weight (lb)")
    plt.scatter(male_x_train[:,0], male_x_train[:,1], c = "blue", s=.1)
    plt.scatter(female_x_train[:,0], female_x_train[:,1], c = "red", s=.1)
    plt.axvline(x = threshold_height, linestyle = 'dashed', color = "black")
    plt.axhline(y = threshold_weight, linestyle = 'dashed', color = "black")
    plt.show()


    #Quadratic Classifier Plot
    ma, mb, mb, mc = male_covariance.ravel()
    fa, fb, fb, fc = female_covariance.ravel()

    xg = np.linspace(50, 80, 1000)
    yg = np.linspace(60, 250, 1000)
    xg, yg = np.meshgrid(xg, yg)

    x3 = (-1/(2*(ma*mc - mb**2))) * ((mc * (xg - male_height_mean)**2) - 2*mb*(xg - male_height_mean) * (yg - male_weight_mean) + (ma * (yg - male_weight_mean)**2))
    y3 = (1/(2*(fa*fc - fb**2))) * ((fc * (xg - female_height_mean)**2) - 2*fb*(xg - female_height_mean)* (yg - female_weight_mean) + (fa * (yg - female_weight_mean) **2))
    d3 = np.log(male_probability/female_probability) - .5 * np.log((ma*mc - mb**2)/(fa*fc- fb**2))
    
    Z = x3 + y3 + d3
    fig0, ax0 = plt.subplots()
    ax0.contour(xg, yg, Z, [0])
    ax0.contourf(xg, yg, Z, levels = 1, cmap = 'RdBu')
    ax0.set_title("Quadratic Classifier")
    ax0.set_xlabel("Height (in)")
    ax0.set_ylabel("Weight (lb)")
    plt.scatter(male_x_train[:,0], male_x_train[:,1], c = "blue", s=.1)
    plt.scatter(female_x_train[:,0], female_x_train[:,1], c = "red", s=.1)
    plt.axvline(x = threshold_height, linestyle = 'dashed', color = "black")
    plt.axhline(y = threshold_weight, linestyle = 'dashed', color = "black")
    plt.show()

    #initial scatter plot for looking at data
    plt.scatter(male_x_train[:,0], male_x_train[:,1], c = "blue", s=.1)
    plt.scatter(female_x_train[:,0], female_x_train[:,1], c = "red", s=.1)
    plt.title("Male vs Female heights and weights")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lb)")
    plt.show()
    return


if __name__ == "__main__":
    main()