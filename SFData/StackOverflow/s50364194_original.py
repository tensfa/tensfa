X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=1)
batch_size = 32
timesteps = None
output_size = 1
epochs=120

inputs = Input(batch_shape=(batch_size, timesteps, output_size))
lay1 = LSTM(20, stateful=True, return_sequences=True)(inputs)
output = Dense(units = output_size)(lay1)
regressor = Model(inputs=inputs, outputs = output)
regressor.compile(optimizer='adam', loss = 'mae')
regressor.summary()

for i in range(epochs):
    print("Epoch: " + str(i))
    regressor.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = batch_size)
    regressor.reset_states()