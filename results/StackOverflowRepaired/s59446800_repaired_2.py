import tensorflow as tf
import numpy as np

x_train = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]],
               dtype="float32")
y_train = np.array([[-1,  1,  3,  5,  7,  9, 11, 13, 15, 17, 19]],
               dtype="float32")

loss = 'mean_squared_error'
optimizer = tf.keras.optimizers.Adam(0.1)

model = tf.keras.Sequential([
tf.keras.layers.Dense(units=11, input_shape=[11])])

model.compile(loss=loss, optimizer=optimizer)

history = model.fit(x_train, y_train, epochs=1)