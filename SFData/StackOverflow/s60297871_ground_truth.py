from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array(np.random.random((7500, 16934)), dtype=np.float32)
y_train = np.random.randint(0, 2, 7500, dtype=np.int32)

classifier = Sequential()
classifier.add(Dense(6,kernel_initializer='random_uniform',activation='relu',input_dim=16934))
classifier.add(Dense(6,kernel_initializer='random_uniform',activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 1)
