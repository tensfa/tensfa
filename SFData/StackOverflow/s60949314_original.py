import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,Flatten,Dense,Dropout,MaxPool2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Activation

dataset = pd.read_csv('fashion-mnist_train.csv')


X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values
X=X/255.
X=np.reshape(X,(-1,28,28,1))
#input_shape_problem_i_think
model=Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=16,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(filters=16,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(units=400,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation='softmax'))
model.summary()
#optimaztion
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X,y,batch_size=100,epochs=1,validation_split=0.33,verbose=1,)