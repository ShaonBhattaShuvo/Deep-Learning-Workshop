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
X,y = make_regression(n_samples=100, n_features=1, n_targets=1, bias=0.5, noise=5.5, random_state=42)

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

#Feature Scaling (Standardization : needs 2D array as input) 
#Here your data Z is rescaled such that μ = 0 and 𝛔 = 1, and is done through this formula: z= (Xi - μ)/𝛔
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y)

# Visulalizing the synthetic dataset after standardization 
print("\nVisualizing the Synthetic Dataset after Standardization")
plt.style.use("ggplot")
plt.scatter(X,y,color='red',edgecolors="green")
plt.title("Synthetic Dataset")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.show()

#Spliting the dataset into Training and Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Creating the deep learning model with hyperbolic tangent activation function 

#A shape (5,2,8) means an array or tensor with 3 dimensions, containing 5 elements in the first dimension, 
#2 in the second and 8 in the third, totaling 30*4*10 = 1200 elements or numbers.
#What flows between layers are tensors. Tensors can be seen as matrices, with shapes.
#In Keras, the input layer itself is not a layer, but a tensor. It's the starting tensor you send to the first 
#hidden layer. This tensor must have the same shape as your training data.
#In our example input data is one dimentional and also has only one element (column). 
model = tf.keras.Sequential([
        tf.keras.Input(shape = (1,)), 
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(1)
        ])
#Compiling the model with Stochatstic Gradient Discent optimizer and MSE as the loss function
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mean_squared_error', metrices=['mean_squared_error'])
#Model's Summary
model.summary()
#Training the model 
model.fit(X_train,y_train, epochs = 50, batch_size =512)
#Testing the models performance 
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
print("Testset Result: \n---------------")
print("MSE: ", mse)

#Reshapping the matrix into array to pass the value into np.linspace() for 2D visualizaiton. 
x_train_arr = np.asarray(X_train).reshape(-1)
y_train_arr = np.asarray(y_train).reshape(-1)
x_test_arr = np.asarray(X_test).reshape(-1)
y_test_arr = np.asarray(y_test).reshape(-1)
#Creating evenly spaced values for smooth visulatization
xp_train = np.linspace(x_train_arr.min(), x_train_arr.max())
xp_test = np.linspace(x_test_arr.min(), x_test_arr.max())

#Visulalizing training and testing plots. 
fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter (X_train, y_train, color='red', edgecolors='green', label='Synthetic Data Points')
ax[0].plot(xp_train,model.predict(xp_train.reshape(-1)),color='blue', label='Regression Line')
ax[0].set_title("tanh Activation Plot (Training Set)")
ax[0].set_xlabel("X_train", fontsize=20)
ax[0].set_ylabel("y_train", fontsize = 20)
ax[0].legend()
ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Synthetic Data Points')
ax[1].plot(xp_test,model.predict(xp_test.reshape(-1)),color='blue',label='Regression Line')
ax[1].set_title("tanh Activation Plot (Testing Set)")
ax[1].set_xlabel("X_test", fontsize=20)
ax[1].set_ylabel("y_test", fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()

#Creating Linear Regression Model
lr_model = LinearRegression()
#Training the model
lr_model.fit(X_train,y_train)
#Testing the model's performance
y_pred_lr = lr_model.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred_lr)
print("MSE: ", mse)

#visualizing the Training and Testset performance of Linear Regression
fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter (X_train, y_train, color='red', edgecolors='green', label='Synthetic Data Points')
ax[0].plot(X_train,lr_model.predict(X_train),color='blue', label='Regression Line')
ax[0].set_title("tanh Activation Plot (Training Set)")
ax[0].set_xlabel("X_train", fontsize=20)
ax[0].set_ylabel("y_train", fontsize = 20)
ax[0].legend()
ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Synthetic Data Points')
ax[1].plot(X_test,lr_model.predict(X_test),color='blue',label='Regression Line')
ax[1].set_title("tanh Activation Plot (Testing Set)")
ax[1].set_xlabel("X_test", fontsize=20)
ax[1].set_ylabel("y_test", fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()
