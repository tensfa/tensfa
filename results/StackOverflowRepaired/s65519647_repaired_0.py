from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import numpy as np

x_train = np.array(np.random.random((3561, 28, 28)), dtype=np.float32)
y_train = np.random.randint(0, 2, 3561, dtype=np.int32)

#Reshape the train data
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
y_train = keras.utils.to_categorical(y_train,2)

# x_train as the folllowing shape (3561, 28, 28, 1)
# y_train as the following shape (3561, 2)

#Build the 2 D CNN model for regression
model= Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(x_train.shape[1],x_train.shape[1:]))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.summary()

#compile the model
model.compile(optimizer='ADADELTA', loss='binary_crossentropy', metrics=['accuracy'])

#train the model

model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=2)