def cnn_model():
    """"
    Creates the model of the CNN.

    :param nr_measures: the number of output nodes needed
    :return: the model of the CNN
    """
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
history = cnn_model.fit(trainX, y_train, batch_size = 50, epochs = 150, verbose = 0 ,validation_data = (validX, y_valid))