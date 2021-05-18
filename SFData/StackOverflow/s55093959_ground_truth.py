from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

train_n = 10
input_n = 8
class_n = 14
train_x = np.array(np.random.random((train_n, input_n)), dtype=np.float32)
train_y = np.random.randint(0, class_n, train_n, dtype=np.int32)
train_y = np.eye(class_n)[train_y]

env_model = Sequential()
env_model.add(Dense(8, activation='relu', input_dim=8))
env_model.add(Dense(128, activation='relu'))
env_model.add(Dense(256, activation='relu'))
env_model.add(Dense(512, activation='relu'))
env_model.add(Dense(14, activation='softmax'))
env_model.summary()

env_model.compile(loss='categorical_crossentropy', optimizer='adam')
env_model.fit(train_x, train_y, epochs=1)
