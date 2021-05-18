from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, Dense
import numpy as np

trainX = np.array(np.random.random((1125, 3841)), dtype=np.float32)
validX = np.array(np.random.random((375, 3841)), dtype=np.float32)
y_train = np.random.randint(0, 2, 1125, dtype=np.int32)
y_valid = np.random.randint(0, 2, 375, dtype=np.int32)

trainX = np.expand_dims(trainX, -1)
validX = np.expand_dims(validX, -1)

def cnn_model(x_train):
    # Initializing the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Conv1D(64, 2, activation="relu", input_shape=(x_train.shape[1], 1)))
    model.add(Flatten())
    model.add(BatchNormalization())

    # Adding the second hidden layer
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())

    # Adding the output layer
    model.add(Dense(1, activation="softplus"))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])

    return model

#Call Model
cnn_model= cnn_model(trainX)

#Train Model
history = cnn_model.fit(trainX, y_train, batch_size = 50, epochs = 1, verbose = 1 ,validation_data = (validX, y_valid))