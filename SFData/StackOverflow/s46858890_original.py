import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
x = np.array([[1],[2],[3],[4],[5]])

y = np.array([[1],[2],[3],[4],[5]])
x_val = np.array([[6],[7]])
x_val = np.array([[6],[7]])
model = Sequential()
model.add(Dense(1, input_dim=5))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x, y, epochs=2, validation_data=(x_val, y_val))