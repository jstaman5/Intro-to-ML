#Jared Staman

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

random_state=425

#-------------------------------------------------------------------------
# The LFW_subset data consists of 100 face images and 100 non-face images. 
# Each image is 25x25. The Haar_like_feature refers to small masks that 
# produce a weighted sum of the pixel values covered by them.
#-------------------------------------------------------------------------

images = lfw_subset()

#-------------------------------------------------------------------------
# Create and display 5x5 grids of the first 25 face and non-face images.
#-------------------------------------------------------------------------

# Your code goes here
fig = plt.figure(figsize=(8,6))
for i in range(25):
  ax = fig.add_subplot(5, 5, i + 1, xticks = [], yticks = [])
  ax.imshow(images[i], cmap = plt.cm.bone)

fig2 = plt.figure(figsize=(8,6))
for i in range(25):
  ax = fig2.add_subplot(5, 5, i + 1, xticks = [], yticks = [])
  ax.imshow(images[i+100], cmap = plt.cm.bone)

#plt.show()
#-------------------------------------------------------------------------
# Select features and size to use (more and bigger is better but slower).
# Extract and store along with face and non-face image labels: X, y.
#-------------------------------------------------------------------------

feature_type2 = ['type-2-x', 'type-2-y'] # 2 masks varying along x, y
feature_type3 = ['type-3-x', 'type-3-y'] # 3 masks varying along x, y
feature_type4 = ['type-4']               # 4 masks varying in x and y

feature_type_set = np.r_[feature_type2, feature_type3, feature_type4]

feature_size = [11,11]

X = [(haar_like_feature(integral_image(images[i]), 
  0, 0, feature_size[0], feature_size[1], feature_type=feature_type_set), i) for i in range(200)]

y = np.concatenate((np.ones(100), np.zeros(100)))
#print(y)  

#-------------------------------------------------------------------------
# Split data into training and test subsets, calculate and apply scaling.
#-------------------------------------------------------------------------
#for i in X:
      #print(i[1])
# Your code goes here
X_train, X_test, y_train, y_test = train_test_split( [i[0] for i in X] , y, test_size=.25, random_state = random_state)

order_train = []
order_test = []
for i in X_train:
      for j in X:
            if np.array_equiv(i, j[0]):
                  order_train.append(j[1])
for i in X_test:
      for j in X:
            if np.array_equiv(i, j[0]):
                  order_test.append(j[1])

#print(order_train)
#print(order_test)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#-------------------------------------------------------------------------
# Use grid search to determine the best set of parameters from given set.
# Train network and apply for test data prediction. Report results.
#-------------------------------------------------------------------------

parameters = [
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'momentum': [0.05, 0.1, 0.9, 0.95],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [random_state],
    'early_stopping': [False],
    'max_iter': [10000],
  }, 
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['tanh'], 
    'solver': ['sgd'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'momentum': [0.05, 0.1, 0.9, 0.95],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [random_state],
    'early_stopping': [False],
    'max_iter': [10000],
  }
]

'''parameters = { 'hidden_layer_sizes': [(16,8)],
    'activation': ['tanh'],
    'solver': ['sgd'],
    'alpha': [0.001],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.1],
    'momentum': [0.1],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [random_state],
    'early_stopping': [False],
    'max_iter': [10000],
  }
'''
# Your code goes here



for i in range(2):

    clf = GridSearchCV(MLPClassifier(), parameters[i], scoring="recall_macro")
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    activation, alpha, early_stop, hidden_layer, learn_rt, learn_rt_in, max_it , momentum, nest_momentum, rand_st, shuffle, solver = clf.best_params_.values()
    best = MLPClassifier(hidden_layer_sizes= hidden_layer, activation= activation, solver= solver, alpha= alpha, learning_rate= learn_rt,\
      learning_rate_init= learn_rt_in, momentum=momentum, nesterovs_momentum=nest_momentum, shuffle=shuffle, random_state=random_state, early_stopping=early_stop,\
        max_iter=max_it ).fit(X_train, y_train)
    y_pred = best.predict(X_test)


    print(best.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    #-------------------------------------------------------------------------
    # Extract network output (MLPClassifier.predict_proba) and plot min/max 
    # ratio with misclassifications indicated by red dots. Report number of 
    # misclassifications. Create and display up to 5 misclassified images.
    #-------------------------------------------------------------------------

    # Your code goes here
    output = best.predict_proba(X_test)
    plt.figure()

    colors = ["blue" if y_pred[x] == y_test[x] else "red" for x in range(len(y_pred))]

    #count the misclassified
    misclassified = 0
    for x in colors:
      if x == "red":
        misclassified += 1

    print(misclassified)

    plt.scatter(np.array([i for i in range(1, len(output) + 1)]) , output[:,1], c = colors)
    #plt.show()


    fig = plt.figure(figsize=(8,6))
 
    count = 0
    count2 = 0
    for i, x in enumerate(X_test):
          if y_pred[i] != y_test[i]:
                count += 1
                ax = fig.add_subplot(1, misclassified, count, xticks = [], yticks = [])
                ax.imshow(images[order_test[i]], cmap = plt.cm.bone)

    plt.show()