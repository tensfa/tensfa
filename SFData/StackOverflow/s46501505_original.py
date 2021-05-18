X_train = numpy.swapaxes(X_train, 1, 3)
X_test = numpy.swapaxes(X_test, 1, 3)

print("X_train shape: ") --> size = (2592, 1, 1366, 96)
print("-----")
print("X_test shape") --> size = (648, 1, 1366, 96)
print("-----")
print(Y_train.shape) --> size = (2592,)
print("-----")
print("Y_test shape") --> size = (648,)

K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

def create_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding="same", input_shape=(1, 1366, 96)))
    model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="th"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model

model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
      batch_size=32,
      epochs=100,
      verbose=1,
      validation_data=(X_test, Y_test))