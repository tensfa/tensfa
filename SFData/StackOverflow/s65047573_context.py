import warnings

import numpy as np

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv1D, Conv2D
from tensorflow.keras.layers import Dense, BatchNormalization

img = np.array(np.random.random((100, 150, 150, 3))).astype(np.float32)
lables = np.random.randint(0, 5, 100, dtype=np.int32)
lables = np.eye(5)[lables]

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

tf.keras.optimizers.Adam(
    learning_rate=0.0001, )
model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])
model.fit(img, lables, batch_size=32, shuffle=True, epochs=1)
