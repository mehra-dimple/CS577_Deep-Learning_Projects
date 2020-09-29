import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
#import tensorflow as tf

df = pd.read_csv('C:\\Users\\dimpl\\Desktop\\communities.data')
arr1=df.to_numpy()
#print(arr1.shape)

#Removing the columns which contains null values
mod_df=df.dropna( axis=1)
arr = mod_df.to_numpy()
print(arr.shape)

input_vector = arr[:,0:102]
print(input_vector.shape)

output_vector = arr[:,102:103]
print(output_vector.shape)

train_data = arr[0:1600,0:102]
test_data = arr[1600:1993,0:102]

train_labels = arr[0:1600,102:103]
test_labels = arr[1600:1993,102:103]

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['mae'])
    return model

k=4
num_val_samples = len(train_data)
num_epochs = 50
all_scores = []
all_mae_histories = []

for i in range(k):
    print('Processing fold #', i)
    val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
    val_targets = train_labels[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i+1)*num_val_samples:]], axis=0)

    model=build_model()
    history=model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,verbose=0)
    mae_history = history.history['val_mean_absolute error']
    all_mae_histories.append(mae_history)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) +1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

np.mean(all_scores)
model=build_model()
model.fit(train_data, train_labels, epochs=20, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)

#model.predict(x_test)
