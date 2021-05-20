from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

X = X.reshape(1,10,1)
y = y.reshape(1,10,1)

data_dim = 1
timesteps = 10

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(10, activation='linear'))

print(model.summary())

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X,y, batch_size=1, epochs=1000)