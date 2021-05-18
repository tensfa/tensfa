from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model =Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

x = np.array(np.random.random((3,1)), dtype=np.float32)

prediction_prob = model.predict(x)