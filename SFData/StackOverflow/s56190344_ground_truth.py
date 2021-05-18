import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense

###
n = 10
n_outputs = 8
n_timesteps = 1452
n_features = 20

# shape
# x_train shape: (1228, 1452, 20)
# y_train shape: (1228, 1452, 8)
# x_val shape: (223, 680, 20)
# x_val shape: (223, 680, 8)
x_train = np.array(np.random.random((n, n_timesteps, n_features)), dtype=np.float32)
y_train = np.random.randint(0, n_outputs, (n, n_timesteps), dtype=np.int32)
y_train = np.eye(n_outputs)[y_train]
y_train = y_train[:,-1,:]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1:]))) # ie 1452, 20
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1,
                batch_size=64,
                shuffle=True)