x, y = load_data(directory)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model = Sequential()
model.add(Convolution3D(1, kernel_size=(3, 3, 3), activation='relu',
                        border_mode='same', name='conv1',
                        input_shape=(50, 50, 50, 1)))
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
    epochs=10,
    batch_size=32,
    )
model.evaluate(x_test, y_test)