# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:48:24 2020

@author: dimpl
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.datasets import make_blobs

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import pandas as pd 
import numpy as np
from matplotlib import pyplot
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
# load dataset
dataset = pd.read_csv('C:\\Users\\dimpl\\Desktop\\iris.data')

#Splitting the data into training and test
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

len(X_test)

model = Sequential()

#Build network

#Input Layer
model.add(Dense(4,input_shape=(4,),activation='relu',activity_regularizer=l1(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Hidden layers
model.add(Dense(8,activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(6,activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Output layer
model.add(Dense(3,activation='softmax'))
model.add(BatchNormalization())

#Compile network
model.compile(optimizer=SGD(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Early Stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# fit the model
history = model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=10,batch_size=5,verbose=0, callbacks=[early_stop])

# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#Plotting
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')