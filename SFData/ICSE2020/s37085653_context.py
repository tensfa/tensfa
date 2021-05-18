import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense

np.random.seed(7)

X = np.array(np.random.random((100, 30, 30)), dtype=np.float32)
Y = np.random.randint(0, 2, 100, dtype=np.int32)

model = Sequential()
model.add(Convolution2D(10, 3, 3, input_shape=(30, 30)))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, Y, epochs=1, batch_size=5)