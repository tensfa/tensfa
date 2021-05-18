import tensorflow as tf
import numpy as np

nb_classes = 3
X_train = np.array(np.random.random((100, 20, 85)), dtype=np.float32)
y_train = np.random.randint(0, nb_classes, 100, dtype=np.int32)
y_train = np.eye(nb_classes)[y_train]
X_test = np.array(np.random.random((10, 20, 85)), dtype=np.float32)
y_test = np.random.randint(0, nb_classes, 10, dtype=np.int32)
y_test = np.eye(nb_classes)[y_test]
batch_size = 32

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, return_sequences=True, stateful=False, input_shape = (20, 85, 1)))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_test, y_test))
