import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(train_images, train_labels, epochs=1)

accuracy = model.evaluate(test_images, test_labels)
print('Accuracy', accuracy)

scores = model.predict(test_images[0:1])
print(np.argmax(scores))