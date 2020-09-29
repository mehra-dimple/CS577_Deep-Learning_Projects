# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:40:38 2020

@author: dimpl
"""

from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import makedirs

from shutil import copyfile
from random import seed
from random import random

import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import layers

import os
  
from keras.preprocessing import image
fnames=[os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fname[3]
img = image.load_img(img_path, target_size=(150,150))
x=image.img_to_array(img)
x=x.reshape((1,)+x.shape)
i=0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plot.show()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(          #train data is augmented 
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
test_datagen = ImageDataGenerator(rescale=1./255)    #test data is not augmented
val_datagen =  ImageDataGenerator(rescale=1./255)    ##validation data is not augmented
#Train_generator
train_generator = train_datagen.flow_from_directory(
    'C:/Users/dimpl/Downloads/dogs-vs-cats/train/train/',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
#Validation_generator
validation_generator = val_datagen.flow_from_directory(
    'C:/Users/dimpl/Downloads/dogs-vs-cats/validation/validation/',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
#Test_generator
test_generator = test_datagen.flow_from_directory(
    'C:/Users/dimpl/Downloads/dogs-vs-cats/test1/test1/',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
    
train_steps_per_epoch = np.math.ceil(train_generator.samples / train_generator.batch_size)
val_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)

result = model.fit_generator(train_generator,
                             steps_per_epoch=train_steps_per_epoch,
                             epochs=40,
                             validation_data=validation_generator,
                             validation_steps=val_steps_per_epoch,
                             workers = 10)
model.save("CatsAndDogs.hdf5")

# define cnn model
def define_model():
    model = Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(200,200,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
   # model.summary()
	# compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_test_harness():
    modelss = define_model()
    datagen = ImageDataGenerator(rescale=1.0/255.0)


    train_it = datagen.flow_from_directory('C:/Users/dimpl/Downloads/dogs-vs-cats/train/train/cats/',
	class_mode='binary', batch_size=20, target_size=(150, 150))
    test_it = datagen.flow_from_directory('C:/Users/dimpl/Downloads/dogs-vs-cats/test1/test1/',
	class_mode='binary', batch_size=20, target_size=(150, 150))

    modelss.fit_generator(train_it, steps_per_epoch=100, epochs=30,
                                  validation_data=test_it, validation_steps=50)
    acc = modelss.evaluate_generator(test_it, steps=50, verbose=0)
    print('> %.3f' % (acc * 100.0))

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

# Test Plot
print("Test Accuracy : ", final_test_acc * 100)
print("Test Loss : ", final_test_loss)

#VGG model
def define_model():
	# load model
    model_frozen = models.Sequential()
	model_frozen = VGG16(include_top=False, input_shape=(150, 150, 3))
	# Freeze the layers
	for layer in model_frozen.layers:
		layer.trainable = True
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model_frozen = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model_frozen.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

import matplotlib.pyplot as plt
hist_dict = model_frozen.history
print(hist_dict.keys())
loss_values = hist_dict['loss']
val_loss_values = hist_dict['val_loss']
epochs = range(1, len(hist_dict['val_loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Frozen_loss.png')

plt.clf()
acc_values = hist_dict['acc']
val_acc_values = hist_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Frozen_acc.png')

# Test Eval
frozen_test_loss, frozen_test_acc = model_frozen.evaluate_generator(
    test_generator,
    steps=test_steps_per_epoch
)
# Test Plot
print("Test Accuracy : ", frozen_test_acc * 100)
print("Test Loss : ", frozen_test_loss)


#Visualization
img_path = 'data1/test/cats/1800.jpg'
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
#print (img_tensor.shape)

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

from keras.models import load_model
model_loaded = load_model('CatsAndDogs.hdf5')
from keras import models
print(model_loaded.layers)
layer_outputs = []
for layer in model_loaded.layers[:8]:
    layer_outputs.append(layer.output)
#layer_outputs = [layer.ouput for layer in model_loaded.layers[:8]]

activation_model = models.Model(inputs=model_loaded.input, outputs= layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
#print (first_layer_activation.shape)

import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4], cmap = 'viridis' )
plt.matshow(first_layer_activation[0,:,:,7], cmap = 'viridis' )

layer_names = []
for layer in model_loaded.layers[:8]:
    layer_names.append(layer.name)
print (layer_names)    

from keras import backend as K
model=load_model('CatsAndDogs.hdf5')  

def deprocess_image(x):
    x-=x.mean()
    x/=(x.std()+1e-5)
    x*=0.1
    x+=0.5
    x=np.clip(x,0,1)
    x*=255
    #Clip to [0,255] and convert to unsigned byte channels
    x=np.clip(x,0,255).astype('uint8')
    return x

def gen_pattern(layer_name,filter_index,size=150):
    layer_output=model.get_layer(layer_name).output
    loss=K.mean(layer_output[:,:,:,filter_index])
    
    grads=K.gradients(loss,model.input)[0]
    grads/=((K.sqrt(K.mean(K.square(grads))))+1e-5)
    iterate=K.function([model.input], [loss,grads])
    input_img_data=np.random.random((1,size,size,3))*20+128
   
    step=1
    for i in range(100):
        loss_value,grads_value=iterate([input_img_data])
        input_img_data+=grads_value*step
       
    img=input_img_data[0]
    return deprocess_image(img)


layer_names=[]
conv_ly1=model.layers[0].name
layer_names.append(conv_ly1)
conv_ly2=model.layers[2].name
layer_names.append(conv_ly2)
conv_ly3=model.layers[4].name
layer_names.append(conv_ly3)

k=0

for layer_name in layer_names:
    layer=model.layers[k]
    layer_output=layer.output
    print(layer_output.shape)
    size=int(layer_output.shape[1])
    row=8
    col=int(layer_output.shape[3]//row)
    results=np.zeros((row*size,col*size,3)).astype('uint0')
   
    for i in range(row):
        for j in range(col):
            filter_img=generate_pattern(layer_name,i+(j*8),size=size)        
            horizontal_start=i*size
            horizontal_end=horizontal_start+size
            vertical_start=j*size
            vertical_end=vertical_start+size
            results[horizontal_start:horizontal_end,vertical_start:vertical_end,:]=filter_img[:,:,:]
    plt.figure(figsize=(20,20))
    plt.title(layer_name)
    plt.imshow(results)
    k+=2

conv_base.trainable = True

model_unfrozen.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

model_unfrozen=model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=val_steps_per_epoch
)
model.save('CatsAndDogs_unfrozen.hdf5')

import matplotlib.pyplot as plt
hist_dict = model_unfrozen.history
print(hist_dict.keys())
loss_values = hist_dict['loss']
val_loss_values = hist_dict['val_loss']
epochs = range(1, len(hist_dict['val_loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Unfrozen_loss.png')

plt.clf()
acc_values = hist_dict['acc']
val_acc_values = hist_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Unfrozen_acc.png')

# Test Eval
unfroz_test_loss, unfroz_test_acc = modelvgg_froz.evaluate_generator(
    test_generator,
    steps=test_steps_per_epoch
)

# Test Plot
print("Test Accuracy : ", unfroz_test_acc)
print("Test Loss : ", unfroz_test_loss)

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

 val_gen = ImageDataGenerator(
     rescale=1./255,
     rotation_range=40,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True
     )

 test_gen = ImageDataGenerator(
     rescale=1./255,
    rotation_range=40,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True
     )
 
#VGG model
def define_model():
	# load model
    model_frozen = models.Sequential()
	model_frozen = VGG16(include_top=False, input_shape=(150, 150, 3))
	# Freeze the layers
	for layer in model_frozen.layers:
		layer.trainable = True
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model_frozen = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model_frozen.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model 
  
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=32,class_mode="binary")

modelvgg_froz.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)
history_vgg_frozen=vgg_froz.fit_generator(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=val_steps_per_epoch
)

vgg_froz.save('cats_and_dogs-frozen-da.hdf5')

import matplotlib.pyplot as plt
hist_dict = history_vgg_frozen.history
print(hist_dict.keys())
loss_values = hist_dict['loss']
val_loss_values = hist_dict['val_loss']
epochs = range(1, len(hist_dict['val_loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Frozen_loss.png')

plt.clf()
acc_values = hist_dict['acc']
val_acc_values = hist_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('Frozen_acc.png')

da_test_loss, da_test_acc = vgg_froz.evaluate_generator(
    test_generator,
    steps=test_steps_per_epoch
)

print("Test Accuracy : ", da_test_acc)
print("Test Loss : ", da_test_loss)
