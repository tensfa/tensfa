import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x = np.array([[1],[2],[3],[4],[5]])

y = np.array([[1],[2],[3],[4],[5]])
x_val = np.array([[6],[7]])
y_val = np.array([[6],[7]])
model = Sequential()
model.add(Dense(1, input_dim=5))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x, y, epochs=1, validation_data=(x_val, y_val))