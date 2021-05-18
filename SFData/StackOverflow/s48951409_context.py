from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

X = np.array(np.random.random((100, 10, 32)), dtype=np.float32)
Y = np.random.randint(0, 10, 100, dtype=np.int32)
Y = np.eye(10)[Y]

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=X.shape))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, Y, batch_size=200, epochs=1, validation_split=0.05)
