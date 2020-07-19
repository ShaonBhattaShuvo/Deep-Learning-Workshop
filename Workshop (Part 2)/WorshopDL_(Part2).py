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

#convert class vectors to binary class matrices
#we can avoid this step by using 'sparse_categorical_crossentropy' loss
#num_classes = len(np.unique(y_test))
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#y_val = keras.utils.to_categorical(y_val, num_classes)

#Creating the deep learning model 
model = tfk.Sequential()
model.add(tfk.layers.Dense(50,input_shape=(784,), activation='relu')) #First Hidden Layer
model.add(tfk.layers.Dense(100, activation='relu')) #Second  Hidden Layer
model.add(tfk.layers.Dense(10, activation='softmax')) #Output Layer

#Model can be crated using following approach as well
#input_units = tfk.Input(shape=(2,))
#hidden_layer1 = tfk.layers.Dense(100, activation ='relu')((input_units))
#hidden_layer2 = tfk.layers.Dense(50, activation ='relu')(hidden_layer1)
#prediction = tfk.layers.Dense(1, activation ='sigmoid')(hidden_layer2)
#model = tfk.models.Model(inputs=input_units, outputs=prediction)

#model Looks like:  2 input -> [50 units in layer1] ->[100 units in layer2] -> 1 output

# Compiling the model for binary classification # Use loss = categorical_crossentropy for multiclass prediction. 
model.compile(optimizer='adam',                         
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
#Model's Summary
model.summary()

#Training the model 
training = model.fit(X_train,y_train, epochs = 20, batch_size =5000, validation_data =(X_val,y_val))

#Visulaizing the Training and Validation Sets Loss and Accuracy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#Plot training and validation accuracy values
#axes[0].set_ylim(0,1) #if we want to limit axis in certain range
axes[0].plot(training.history['sparse_categorical_accuracy'], label='Train')
axes[0].plot(training.history['val_sparse_categorical_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
#Plot training and validation loss values
#axes[1].set_ylim(0,1)
axes[1].plot(training.history['loss'], label='Train')
axes[1].plot(training.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.tight_layout()
plt.show()

# Evaluating the performance on the Test set 
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
 

