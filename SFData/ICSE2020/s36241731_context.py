from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(keras.layers.SimpleRNN(100))
model.compile(loss="mse", optimizer="sgd")
X_train = np.random.rand(119,80)
y_train = np.array([[1]])
model.fit(X_train, y_train, batch_size=32)