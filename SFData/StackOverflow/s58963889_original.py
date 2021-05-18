from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

df = pd.read_csv("C:\\Users\\Dan\\y_sinx.csv")

x = df['x'].values #Pandas to Numpy
y = df['y'].values


print(type(x)) #check type
print(np.shape(x)) #check dimensions
print(x) #check x

#Network
model = Sequential()
model.add(Dense(7, input_shape = x.shape, activation='relu')) #Hidden layer 1
model.add(Dense(4, activation='relu')) #Hidden layer 2
model.add(Dense(1)) #Output layer
model.compile(loss='mean_squared_error', optimizer = 'adam')
model.fit(x, y, verbose = 2, epochs = 20)