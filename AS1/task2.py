import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

df = pd.read_csv('C:\\Users\\dimpl\\Desktop\\spambase.data')
arr = df.to_numpy()
  
input_vector = arr[:,0:57]
print(input_vector.shape)

output_vector = arr[:,57:58]
print(output_vector.shape)

train_set = arr[0:3000,0:57].astype('float32')/255
test_set = arr[3000:4600,0:57].astype('float32')/255

train_labels = arr[0:3000,57:58].astype('float32')/255
test_labels = arr[3000:4600,57:58].astype('float32')/255


def to_one_hot(labels,dimension=1000):
    results=np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i,label]=1
        return results

model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(train_set.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_set,train_labels,epochs=20,batch_size=100)

results = model.evaluate(test_set, test_labels)
print(model.metrics_names)
print('Test result: ', results)
