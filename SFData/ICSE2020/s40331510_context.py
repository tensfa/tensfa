from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
import numpy as np


X = np.array(np.random.random((2, 10, 10)), dtype=np.float32)
Y = np.random.randint(0, 2, 2, dtype=np.int32)
Y = np.eye(2)[Y]

model = Sequential()
model.add(LSTM(100, input_shape =(10,10)))
model.add(LSTM(100))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1, batch_size=1)