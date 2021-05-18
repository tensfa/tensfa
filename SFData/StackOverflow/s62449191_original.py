import numpy as np

#Install Tensor Flow
try:
  #Tensorflow_version solo existe en Colab
  %tensorflow_version 2.x

except Exception:
  pass

import tensorflow as tf

tf.__version__

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='Greys');

y_train[0]

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train.shape

model = tf.keras.Sequential([
                           tf.keras.layers.Flatten(input_shape=(28, 28)),
                           tf.keras.layers.Dense(units=512, activation='relu'),
                           tf.keras.layers.Dense(units=10, activation='softmax')
])
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=10, batch_size=256)