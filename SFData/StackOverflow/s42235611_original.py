from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Cropping2D(cropping=((0,0), (50,20)), input_shape=(160 ,320, 3))) #(None, 90, 320, 3)
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(32, 3, 3,)) #(None, 88, 318, 32)
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3)) #(None, 86, 316, 32)
model.add(Activation('relu'))
model.add(Flatten()) #(None, 869632)
model.add(Dense(128)) #(None, 128)
model.add(Activation('relu'))
model.add(Dense(4)) #(None, 4)
print(model.summary())

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, validation_split=0.2, batch_size=32, nb_epoch=3, verbose=1)