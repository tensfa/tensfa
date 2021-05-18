import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten


reshape_channel_train = loadmat('reshape_channel_train')
reshape_channel_test = loadmat('reshape_channel_test.mat')
reshape_label_train = loadmat('reshape_label_train')
reshape_label_test = loadmat('reshape_label_test')

X_train = reshape_channel_train['store_train']
X_test = reshape_channel_test['store_test']

X_train = np.expand_dims(X_train,axis = 0)
X_test  = np.expand_dims(X_test, axis = 0)

Y_train = reshape_label_train['label_train']
Y_test = reshape_label_test['label_test']

classifier = Sequential()
classifier.add(Conv2D(8, kernel_size=(3,3) , input_shape=(3, 12, 1), padding="same"))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))

classifier.add(Conv2D(8, kernel_size=(3,3), input_shape=(3, 12, 1), padding="same"))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))

classifier.add(Flatten())
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(6, activation='sigmoid'))
classifier.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

history = classifier.fit(X_train, Y_train, batch_size = 32, epochs=100,
                         validation_data=(X_test, Y_test), verbose=2)