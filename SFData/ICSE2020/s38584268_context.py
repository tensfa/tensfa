from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
import numpy as np

X = np.array(np.random.random((10, 100, 100, 3)), dtype=np.float32)
Y = np.random.randint(0, 50, 10, dtype=np.int32)
Y = np.eye(50)[Y]
input_img_shape = (100, 100, 3)

model = Sequential()

#C1
model.add(Convolution2D(15, 7, 7, activation='relu', input_shape=input_img_shape))
print("C1 shape: ", model.output_shape)

#S2
model.add(MaxPooling2D((2,2)))
print("S2 shape: ", model.output_shape)
#...

#C5
model.add(Convolution2D(250, 5, 5, activation='relu'))
print("C5 shape: ", model.output_shape)

#F6
model.add(Dense(50))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1, batch_size=1)