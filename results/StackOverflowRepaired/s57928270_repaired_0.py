from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution3D, MaxPooling3D, Flatten, Dense
import numpy as np

x = np.array(np.random.random((10, 50, 50, 50 ,1)), dtype=np.float32)
y = np.random.randint(0, 32, 10, dtype=np.int32)
y = np.eye(32)[y]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Convolution3D(1, kernel_size=(3, 3, 3), activation='relu',
                        padding='same', name='conv1', input_shape=(50, 50, 50, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']
    )
model.fit(
    x_train,
    y_train,
    epochs=1,
    batch_size=32,
    )
model.evaluate(x_test, y_test)