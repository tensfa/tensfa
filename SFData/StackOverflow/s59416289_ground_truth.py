from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Dropout, BatchNormalization, Flatten
import numpy as np

Xtrain = np.array(np.random.random((100, 200, 200, 3)), dtype=np.float32)
Ytrain = np.random.randint(0, 2, 100, dtype=np.int32)
Ytrain = np.eye(2)[Ytrain]
Xtest = np.array(np.random.random((10, 200, 200, 3)), dtype=np.float32)
Ytest = np.random.randint(0, 2, 10, dtype=np.int32)
Ytest = np.eye(2)[Ytest]

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=Xtrain.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2, activation='relu'))

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest))
