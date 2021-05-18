import tensorflow as tf
from tensorflow.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.Sequential([
                           tf.keras.layers.Flatten(input_shape=(28, 28)),
                           tf.keras.layers.Dense(units=512, activation='relu'),
                           tf.keras.layers.Dense(units=10, activation='softmax')
])
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=1, batch_size=256)