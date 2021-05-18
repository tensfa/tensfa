model = Sequential()
model.add(LSTM(100,input_shape =(time_steps,vector_size)))
model.add(LSTM(100))