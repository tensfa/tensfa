from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation
import numpy as np

X_train = np.random.normal(size=(100,150))
Y_train = np.random.randint(0,9,size=100)

num_classes=9
Y_train = keras.utils.to_categorical(Y_train, num_classes)
#Reshape data to add new dimension
X_train = X_train.reshape((100, 150, 1))
Y_train = Y_train.reshape((100, 9))
model = Sequential()
model.add(Conv1D(1, kernel_size=3, activation='relu', input_shape=(150, 1)))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=X_train,y=Y_train, epochs=1, batch_size=20)