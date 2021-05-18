import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, Activation, MaxPooling1D, Flatten, Dense

np.random.seed(7)

X = np.array(np.random.random((100, 101, 1)), dtype=np.float32)
Y = np.random.randint(0, 4, 100, dtype=np.int32)
Y = np.eye(4)[Y]

conv = Sequential()
conv.add(Convolution1D(64, 10, input_shape=(101, 1)))
conv.add(Activation('relu'))
conv.add(MaxPooling1D(2))
conv.add(Flatten())
conv.add(Dense(10))
conv.add(Activation('tanh'))
conv.add(Dense(4))
conv.add(Activation('softmax'))

conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
conv.fit(X, Y, epochs=1, batch_size=5)