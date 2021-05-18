from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X_train = np.array(np.random.random((180, 10)), dtype=np.float32)
y_train = np.random.randint(0, 2, 180, dtype=np.int32)

model = Sequential()
model.add(Dense(100, activation='relu', input_shape = X_train.shape[1:]))
model.add(Dense(500, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 1)
