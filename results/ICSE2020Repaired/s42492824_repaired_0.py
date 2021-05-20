from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import numpy as np


X = np.array(np.random.random((60, 60)), dtype=np.float32)
Y = np.random.randint(0, 2, 60, dtype=np.int32)


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=[60]))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(122, activation='softmax'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

model = create_baseline()
model.fit(X, Y, epochs=1, batch_size=1)