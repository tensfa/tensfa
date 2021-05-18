import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

n = 10
data = np.random.randint(0,256,(n, 100, 100, 3))
data = np.array(data, dtype="float") / 255.0
labels = np.random.randint(0, 2, n, dtype=np.int32)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)

print(trainX.shape)
print(trainY.shape)

print('done')

model = Sequential()
model.add(Convolution2D(32, (3,3), input_shape = (100,100,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Convolution2D(64, (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Convolution2D(128, (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Convolution2D(256, (1,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'relu'))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#train the network
print("[INFO] training network...")
hist = model.fit(trainX, trainY, batch_size=256, epochs=1, validation_split=0.3)