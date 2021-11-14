import numpy as np

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

random_state=525

#-------------------------------------------------------------------------
# The LFW_subset data consists of 100 face images and 100 non-face images. 
# Each image is 25x25. The Haar_like_feature refers to small masks that 
# produce a weighted sum of the pixel values covered by them.
#-------------------------------------------------------------------------

images = lfw_subset()
print(images)
#-------------------------------------------------------------------------
# Create and display 5x5 grids of the first 25 face and non-face images.
#-------------------------------------------------------------------------

# Your code goes here

#-------------------------------------------------------------------------
# Select features and size to use (more and bigger is better but slower).
# Extract and store along with face and non-face image labels: X, y.
#-------------------------------------------------------------------------

feature_type2 = ['type-2-x', 'type-2-y'] # 2 masks varying along x, y
feature_type3 = ['type-3-x', 'type-3-y'] # 3 masks varying along x, y
feature_type4 = ['type-4']               # 4 masks varying in x and y

feature_type_set = np.r_[feature_type2, feature_type3, feature_type4]

feature_size = [11,11]

X = haar_like_feature(integral_image(images[0]), 
  0, 0, feature_size[0], feature_size[1], feature_type=feature_type_set)

#-------------------------------------------------------------------------
# Split data into training and test subsets, calculate and apply scaling.
#-------------------------------------------------------------------------

# Your code goes here

#-------------------------------------------------------------------------
# Use grid search to determine the best set of parameters from given set.
# Train network and apply for test data prediction. Report results.
#-------------------------------------------------------------------------

parameters = [
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'momentum': [0.1, 0.9],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [random_state],
    'early_stopping': [False],
    'max_iter': [10000],
  }, 
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['tanh'], 
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'momentum': [0.1, 0.9],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [random_state],
    'early_stopping': [False],
    'max_iter': [10000],
  }
]

# Your code goes here

#-------------------------------------------------------------------------
# Extract network output (MLPClassifier.predict_proba) and plot min/max 
# ratio with misclassifications indicated by red dots. Report number of 
# misclassifications. Create and display up to 5 misclassified images.
#-------------------------------------------------------------------------

# Your code goes here
