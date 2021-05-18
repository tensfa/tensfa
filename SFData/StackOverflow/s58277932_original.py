num_classes=9
Y_train = keras.utils.to_categorical(Y_train, num_classes)
#Reshape data to add new dimension
X_train = X_train.reshape((100, 150, 1))
Y_train = Y_train.reshape((100, 9, 1))
model = Sequential()
model.add(Conv1d(1, kernel_size=3, activation='relu', input_shape=(None, 1)))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=X_train,y=Y_train, epochs=200, batch_size=20)