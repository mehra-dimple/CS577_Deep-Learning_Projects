# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:00:05 2020

Assignment 4 - CIFAR-10
"""

from numpy import *
# import cv2 as cv
from time import sleep
import os
import errno
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
import pickle

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred



def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def read_data(folder):
    x_data_temp = []
    y_data_temp = []
    x_test_data_temp = []
    y_test_data_temp = []

    for file in os.listdir(folder):
        if file.endswith(".meta") or file.endswith(".html"):
            print("Ignoring html and meta files")
        elif "test_batch" in file:
            # test data file detected. we are gonna load it separately
            test_data_temp = unpickle(folder + "/" + file)
            x_test_data_temp.append(test_data_temp[b'data'])
            y_test_data_temp.append(test_data_temp[b'labels'])
        else:
            temp_data = unpickle(folder + "/" + file)
            x_data_temp.append(temp_data[b'data'])
            y_data_temp.append(temp_data[b'labels'])
    x_data = array(x_data_temp)
    y_data = array(y_data_temp)
    x_test_data = array(x_test_data_temp)
    y_test_data = array(y_test_data_temp)
    return [x_data, y_data, x_test_data, y_test_data]


X_train_temp, y_train_temp, X_test_temp, y_test_temp = read_data("cifar-10-batches-py")

# At this time, since we converted from list to numpy array, there ia an extra dimension added to the array
# X_train_temp.shape = (6, 10000, 3072) and y_train_temp.shape = (6, 10000)
# In order to fix this, we will need to reshape the stack.

X_train_temp = X_train_temp.reshape(X_train_temp.shape[0] * X_train_temp.shape[1], X_train_temp.shape[2])
y_train_temp = y_train_temp.reshape(y_train_temp.shape[0] * y_train_temp.shape[1])

# Similarly for X_test_temp and y_test_data

X_test_temp = X_test_temp.reshape(X_test_temp.shape[0] * X_test_temp.shape[1], X_test_temp.shape[2])
y_test_temp = y_test_temp.reshape(y_test_temp.shape[0] * y_test_temp.shape[1])

print(X_train_temp.shape, X_train_temp.ndim, type(X_train_temp))
print(y_train_temp.shape, y_train_temp.ndim, type(y_train_temp))

print(X_test_temp.shape, X_test_temp.ndim, type(X_test_temp))
print(y_test_temp.shape, y_test_temp.ndim, type(y_test_temp))

# Now lets shuffle the data a bit with random state 4

X_train, y_train = shuffle(X_train_temp, y_train_temp, random_state=4)
X_test, y_test = shuffle(X_test_temp, y_test_temp, random_state=4)

# Splitting X and y in training and val data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

# Keras Parameters
batch_size = 32
nb_classes = 10
nb_epoch = 20
img_rows, img_col = 32, 32
img_channels = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

# Now that our data has been shuffled and spitted,  lets reshape it and get it ready to be fed into our CCN model

X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
y_train = np_utils.to_categorical(y_train, nb_classes)

X_val = X_val.reshape(X_val.shape[0], 3, 32, 32)
y_val = np_utils.to_categorical(y_val, nb_classes)

X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Finally print shape of this data :

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Regularize the data
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_val /= 255
X_test /= 255

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import add
from keras.models import Model
from tensorflow import  keras


# Build the sequential model
inputShape = Input(shape = (32,32,3))

layer1 = Conv2D(32,(3,3),padding='same',activation='relu')(inputShape)
layer1 = MaxPooling2D(pool_size=(2,2))(layer1)
layer1 = Dropout(0.5)(layer1)

layer2 = Conv2D(64,(3,3),padding='same',activation='relu')(layer1)
layer2 = MaxPooling2D(pool_size=(2,2))(layer2)
layer2 = Dropout(0.5)(layer2)

layer3 = Conv2D(64,(3,3),padding='same',activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=(2,2))(layer3)
layer3 = Dropout(0.5)(layer3)

output = Flatten()(layer3)
output = Dense(64,activation='relu')(output)
out = Dense(10,activation='softmax')(output)

model = Model(inputs = inputShape,outputs = out)

# Compile the model
adam = Adam(lr=0.001)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

trainGen = ImageDataGenerator()
result = model.fit_generator(trainGen.flow(trainX,trainY,batch_size = 512),
                             validation_data = (valX,valY),
                             epochs=60,
                             steps_per_epoch= len(trainX)//512,
                             workers=20)

# Plot the training and validation loss
import matplotlib.pyplot as plt
loss = result.history['loss']
val_loss = result.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss') # blue dots
plt.plot(epochs,val_loss,'b',label='Validation loss') # blue line
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the training and validation accuracy
plt.clf()
acc_values = result.history['accuracy']
val_acc_values = result.history['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predict = model.evaluate(testX,testY,verbose=1)
print("Test Accuracy :",predict[1])
print("Test Loss :",predict[0])

# Save the model
model.save('CIFAR10.hdf5')

# Make a prediction
model.predict(testX[10:11])

# Build the sequential model with inception block
inputShape = Input(shape = (32,32,3))

layer1 = Conv2D(32,(3,3),padding='same',activation='relu')(inputShape)
layer1 = MaxPooling2D(pool_size=(2,2))(layer1)

layer2 = Conv2D(64,(3,3),padding='same',activation='relu')(layer1)
layer2 = MaxPooling2D(pool_size=(2,2))(layer2)

layer3 = Conv2D(64,(3,3),padding='same',activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=(2,2))(layer3)

layer4 = Conv2D(64,(3,3),padding='same',activation='relu')(layer3)
layer4 = BatchNormalization()(layer4)
layer4 = MaxPooling2D(pool_size=(2,2))(layer4)
layer4 = Dropout(0.5)(layer4)

block1_1 = Conv2D(64, (1,1), activation='relu', padding='same')(layer4)
block1_1 = Conv2D(64, (3,3), activation='relu', padding='same')(block1_1)
block1_1 = Dropout(0.5)(block1_1)

block1_2 = MaxPooling2D((3,3), strides = (1,1), padding='same')(layer4)
block1_2 = Conv2D(64, (1,1), activation='relu', padding='same')(block1_2)
block1_2 = Dropout(0.5)(block1_2)

block1 = concatenate([block1_1, block1_2], axis=3)
block1 = BatchNormalization()(block1)

output = Flatten()(block1)
out = Dense(10,activation='softmax')(output)

model = Model(inputs = inputShape,outputs = out)

model.summary()

# Compile the model
adam = Adam(lr=0.001)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the model with 1000 epochs
result = model.fit_generator(trainGen.flow(trainX,trainY,batch_size = 512),
                             validation_data = (valX,valY),
                             epochs = 60,
                             steps_per_epoch= len(trainX)//512,
                             workers = 20)
results=model.evaluate(testX,testY)
print(results)

# Plot the training and validation loss
import matplotlib.pyplot as plt
loss = result.history['loss']
val_loss = result.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss') # blue dots
plt.plot(epochs,val_loss,'b',label='Validation loss') # blue line
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the training and validation accuracy
plt.clf()
acc_values = result.history['accuracy']
val_acc_values = result.history['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predict = model.evaluate(testX,testY,verbose=1)
print("Test Accuracy :",predict[1])
print("Test Loss :",predict[0])

model.save('CIFAR10_inception.hdf5')
