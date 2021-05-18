model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=X.values.shape))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X.values, Y.values, batch_size=200, epochs=10, validation_split=0.05)