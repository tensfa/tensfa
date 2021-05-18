import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

look_back = 3
trainX = np.array(np.random.random((100, look_back, 3)), dtype=np.float32)
trainY = np.array(np.random.random(100), dtype=np.float32)
testX = np.array(np.random.random((10, look_back, 3)), dtype=np.float32)
testY = np.array(np.random.random(10), dtype=np.float32)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history= model.fit(trainX, trainY,validation_split=0.33, nb_epoch=1, batch_size=32)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)