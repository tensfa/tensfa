from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten

data = np.array(np.random.random((11200, 5, 54)), dtype=np.float32)
target = np.random.randint(0, 2, 11200, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
batch_size = 32
timesteps = None
output_size = 1
epochs=1

inputs = Input(batch_shape=(batch_size, timesteps, output_size))
lay1 = LSTM(20, stateful=True, return_sequences=True)(inputs)
output = Flatten()(lay1)
output = Dense(units = output_size)(output)
regressor = Model(inputs=inputs, outputs = output)
regressor.compile(optimizer='adam', loss = 'mae')
regressor.summary()

for i in range(epochs):
    print("Epoch: " + str(i))
    regressor.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = batch_size)
    regressor.reset_states()