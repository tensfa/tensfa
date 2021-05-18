from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# load dataset
dataset = np.loadtxt("data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = Sequential()
model.add(Dense(8, activation="relu", input_dim=8, kernel_initializer="uniform"))
model.add(Dense(12, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
test = np.array([6,148,72,35,0,33.6,0.627,50])
predictions = model.predict(test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)