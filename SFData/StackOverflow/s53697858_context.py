from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X_train = np.array(np.random.random((352, 18)), dtype=np.float32)
y_train = np.random.randint(0, 6, 352, dtype=np.int32)
y_train = np.eye(6)[y_train]
X_test = np.array(np.random.random((152, 18)), dtype=np.float32)

classifier = Sequential()
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6 ,kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)
y_pred = classifier.predict(X_test)