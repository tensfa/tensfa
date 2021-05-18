import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

batch_size = 32
class_num = 3
IMG_WIDTH = 192
IMG_HEIGHT = 192

X = np.random.sample((batch_size, IMG_WIDTH, IMG_HEIGHT, 3))
y = np.eye(class_num)[np.random.randint(0, class_num, batch_size)]
y = np.argmax(y, -1)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
for i in range(2):
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=1)
