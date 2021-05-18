
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (X_train.shape[0],)))
model.add(Dense(500, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 200)