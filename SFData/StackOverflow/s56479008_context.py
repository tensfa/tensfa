from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
import numpy as np

x_train = np.array(np.random.random((128, 64, 64, 1)), dtype=np.float32)
x_test = np.array(np.random.random((64, 64, 64, 1)), dtype=np.float32)
y_train = np.random.randint(0, 24, 128, dtype=np.int32)
y_test = np.random.randint(0, 24, 64, dtype=np.int32)

model = Sequential()

model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64, 64, 1), padding='same'))
model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu', padding='same'))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(24, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1)