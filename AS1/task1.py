
from keras.datasets import cifar10
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
##from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
from keras.utils import to_categorical

#Load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
##(training_img, training_labels), (testing_images, testing_labels) = cifar10.load_data()
##train_images, test_images, train_labels, test_labels = train_test_split(training_img, testing_images, test_size=0.33)

print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(train_images.shape)
print(len(test_labels))
print(test_labels)

#Create validation set 
train_images = train_images.reshape((50000, 32*32*3))
test_images = test_images.reshape((10000, 32*32*3))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build network
input_tensor = layers.Input(shape=(784,))
x=layers.Dense(32, activation='relu')(input_tensor)
output_tensor=layers.Dense(10,activation='softmax')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)

#Vectorize into data and labels
def vectorize_sequences(sequences, dimension=18000):
    results=np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
        return results
    
x_train=vectorize_sequences(train_images)
x_test=vectorize_sequences(test_images)
y_train=np.asarray(train_labels).astype('float32')/255
y_test=np.asarray(test_labels).astype('float32')/255

#Adding hidden layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(32*32*3,)))
model.add(layers.Dense(16, activation='relu'))

#OUTPUT LAYER
model.add(layers.Dense(10, activation='softmax'))

#Compile the model
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])


history=model.fit(train_images, train_labels, epochs=20, batch_size=512)
history_dict=history.history
print(history_dict)

#Training and validation
import matplotlib.pyplot as plt
history_dict=history.history

print(history_dict)
val_loss_values   = history_dict['loss']
val_acc_values   = history_dict['binary_accuracy']

epochs=range(1,len(val_acc_values)+1)
 

plt.plot(epochs, val_loss_values, 'b', 'Training loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, val_acc_values, 'b', label='Training accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Evaluate the model
test_loss, test_acc=model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


