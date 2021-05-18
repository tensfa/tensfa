# shape
# x_train shape: (1228, 1452, 20)
# y_train shape: (1228, 1452, 8)
# x_val shape: (223, 680, 20)
# x_val shape: (223, 680, 8)

###
n_outputs = 8
n_timesteps = 1452
n_features = 20

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1:]))) # ie 1452, 20
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=9,
                batch_size=64,
                shuffle=True)