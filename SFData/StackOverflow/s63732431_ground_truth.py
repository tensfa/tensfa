import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import numpy as np

cnn = Sequential()
num_timesteps = 2

# 1st conv layer
cnn.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48, 1)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

# 2nd conv layer
cnn.add(Conv2D(128,(5,5), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

# 3rd conv layer
cnn.add(Conv2D(512,(3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

# 4th conv layer
cnn.add(Conv2D(512,(3,3), padding='same'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

# flatten
cnn.add(Flatten())

# fully connected 1
cnn.add(Dense(256))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(0.5))

#fully connected 2
cnn.add(Dense(512))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(0.5))


model = Sequential()
model.add(Reshape((1, 48, 48, 1)))
model.add(TimeDistributed(cnn, input_shape=(1, 48, 48, 1)))
model.add(LSTM(num_timesteps))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

model.fit(np.random.rand(100, 48, 48, 1), np.random.rand(100))