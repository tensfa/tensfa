from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
import numpy as np


X = np.array(np.random.random((10, 512, 64, 3)), dtype=np.float32)
Y = np.random.randint(0, 2, 10, dtype=np.int32)
Y = np.eye(2)[Y]

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape =(512,64,3)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1, batch_size=1)
image = np.array(np.random.random((512, 64, 3)), dtype=np.float32)
image = np.expand_dims(image, 0)
model.predict(image)