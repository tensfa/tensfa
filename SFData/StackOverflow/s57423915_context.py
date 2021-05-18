import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv1D(1, 5, input_shape=(6,1), activation="tanh"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# df is pandas DataFrame
X = np.array(np.random.random((4725, 6)), dtype=np.float32)
y = np.array(np.random.randint(0, 2, 4725), dtype=np.int32)
y = np.eye(2)[y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
mode = baseline_model()
history = mode.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))