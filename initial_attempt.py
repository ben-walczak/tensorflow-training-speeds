# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:11:26 2019

@author: Benjamin
"""

import pandas as pd
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

is_gpu_version = False
gpu_count = 0
raw_devices = device_lib.list_local_devices()
print(raw_devices)

for device in raw_devices:
    if device.device_type == 'GPU':
        is_gpu_version = True
        gpu_count += 1
        
print("Number of GPU's: "+str(gpu_count))




df = pd.read_csv('winemag-data_first150k.csv')  

# unnecessary id
del df['Unnamed: 0']
del df['description']
# too many categorical variables
del df['designation']
del df['province']
del df['region_1']
del df['winery']

df = pd.get_dummies(df,['country','region_2','variety'])
df = df.dropna()

train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)
train_labels = train_dataset.pop('points')
test_labels = test_dataset.pop('points')

def build_model():
    input_shape=(len(train_dataset.keys()),)
    
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

model.summary()

example_batch = train_dataset[:10]
example_result = model.predict(example_batch)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 30

start = time.time()

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

end = time.time()

total_time = round(end - start,3)

print()
print(end - start)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Wine Rating^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.title(str(gpu_count)+' GPU(s) used to train the model in '+str(total_time)+ ' seconds')
  plt.legend()


plot_history(history)

plt.savefig(str(gpu_count)+'_'+str(total_time)+'_'+str(EPOCHS)+'.png')