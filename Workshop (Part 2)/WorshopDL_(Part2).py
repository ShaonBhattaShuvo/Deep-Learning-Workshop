# -*- coding: utf-8 -*-
"""
@author: Shaon Bhatta Shuvo
"""
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skd
from sklearn import preprocessing 
import tensorflow.keras as tfk
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from mlxtend.plotting import plot_decision_regions

#loading the datset from keras 
mnist = tfk.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data() #X indiciates images and y indicates lables 

#Basic dataset info
print("Number of distict labels or classes: ", np.unique(y_test))
print("Number of train images: ",len(y_train))
print("Number of test images: ",len(y_test))

#some sample dataset images
print("Dataset Samples: ")
class_names = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i],)
    plt.xlabel(class_names[y_train[i]])
plt.show()

#Data Rescaling: 
#All the images are consist of RGB values with pixel range from 0 to 255. However, we need to rescale 
#these values between 0 to 1, because values in range 0 to 255 would be too high for our model to process.
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train /= 255.0
X_test /= 255.0

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Reserve 10,000 samples for Validation Set
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]

 

