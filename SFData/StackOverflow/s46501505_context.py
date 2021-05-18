from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np


X_train = np.array(np.random.random((100, 1366, 96, 1)), dtype=np.float32)
Y_train = np.random.randint(0, 2, 100,dtype=np.int32)

X_test = np.array(np.random.random((10, 1366, 96, 1)), dtype=np.float32)
Y_test = np.random.randint(0, 2, 10,dtype=np.int32)

def create_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same", input_shape=(1366, 96, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model

model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
      batch_size=32,
      epochs=1,
      verbose=1,
      validation_data=(X_test, Y_test))