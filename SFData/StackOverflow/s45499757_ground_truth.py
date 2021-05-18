from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# load dataset
X = np.array(np.random.random((100, 8)), dtype=np.float32)
Y = np.random.randint(0, 2, 100, dtype=np.int32)

# create model
model = Sequential()
model.add(Dense(8, activation="relu", input_dim=8, kernel_initializer="uniform"))
model.add(Dense(12, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=1, batch_size=10,  verbose=2)

# calculate predictions
test = np.array([6,148,72,35,0,33.6,0.627,50])
test = np.expand_dims(test, 0)

predictions = model.predict(test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)