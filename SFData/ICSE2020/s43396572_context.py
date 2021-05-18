import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

X = np.array(np.random.random((100, 30)), dtype=np.float32)
Y = np.array(np.random.randint(0, 2, 100), dtype=np.int32)

X_test = np.array(np.random.random((100, 30)), dtype=np.float32)
Y_test = np.array(np.random.randint(0, 2, 100), dtype=np.int32)

model = Sequential()
model.add(Conv1D(2,2,activation='relu',input_shape=(30,1)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1, batch_size=5)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))