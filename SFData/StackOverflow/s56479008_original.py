model = Sequential()

model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64, 64, 1), padding='same'))
model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu', padding='same'))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(24, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=8)