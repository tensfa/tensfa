import numpy as np
from tensorflow.keras.layers import Dense, LSTM, AveragePooling1D, Flatten
from tensorflow.keras.models import Sequential

trainX = np.array(np.random.random((100, 50)), dtype=np.float32)
trainX = np.expand_dims(trainX, -1)
trainY = np.array(np.random.random(100), dtype=np.float32)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(50,1), activation='tanh',recurrent_activation='sigmoid',return_sequences=True))
model.add(AveragePooling1D(pool_size=2, strides=2,padding='valid'))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history= model.fit(trainX, trainY,validation_split=0.33, nb_epoch=1, batch_size=32)
