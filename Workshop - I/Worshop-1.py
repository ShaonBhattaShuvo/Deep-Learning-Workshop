# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:07:36 2020

@author: Shaon Bhatta Shuvo
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn import preprocessing 
import tensorflow as tf
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics 

#Generating synthetic linear dataset
X,y = make_regression(n_samples=100, n_features=1, n_targets=1, 
                                 bias=0.5, effective_rank=None, tail_strength=0.5, noise=5.5, 
                                 shuffle=True, coef=False, random_state=None)

# Visulalizing the synthetic dataset
print("\nVisualizing the Synthetic Dataset")
plt.style.use("ggplot")
plt.scatter(X,y,color='red',edgecolors="green")
plt.title("Synthetic Dataset")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.show()

#reshaping the y values into 2D matrix of 1 column
y = y.reshape(-1,1) #if y is not an array then use, np.asanyarray(y).reshape(-1,1)
# Equivalent code y = np.reshape(y,(-1,1))

#Feature Scaling (Standardization : needs matrix as input) 
#Here your data Z is rescaled such that μ = 0 and 𝛔 = 1, and is done through this formula: z= (Xi - μ)/𝛔
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y)

# Visulalizing the synthetic dataset after standardization 
print("\nVisualizing the Synthetic Dataset")
plt.style.use("ggplot")
plt.scatter(X,y,color='red',edgecolors="green")
plt.title("Synthetic Dataset")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.show()

