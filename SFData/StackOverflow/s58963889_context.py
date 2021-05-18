import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

x = np.array(np.random.random((7, 1)), dtype=np.float32)
y = np.array(np.random.random((7, 1)), dtype=np.float32)

#Network
model = Sequential()
model.add(Dense(7, input_shape = x.shape, activation='relu')) #Hidden layer 1
model.add(Dense(4, activation='relu')) #Hidden layer 2
model.add(Dense(1)) #Output layer
model.compile(loss='mean_squared_error', optimizer = 'adam')
model.fit(x, y, verbose = 2, epochs = 1)