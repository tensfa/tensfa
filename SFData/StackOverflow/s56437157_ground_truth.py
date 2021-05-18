from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import numpy as np

x_train = np.random.random((30,50,50,3))
y_train = np.random.randint(2, size=(30,1))

model = Sequential()

#start from the first hidden layer, since the input is not actually a layer
#but inform the shape of the input, with 3 elements.
model.add(Dense(units=4,input_shape=(50,50,3))) #hidden layer 1 with input

#further layers:
model.add(Dense(units=4)) #hidden layer 2
model.add(Flatten())
model.add(Dense(units=1)) #output layer

model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

model.fit(x_train, y_train,
       epochs=1,
       batch_size=128)