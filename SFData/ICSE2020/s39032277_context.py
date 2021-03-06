import tensorflow as tf
import numpy as np
import random
import os
import cv2
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(os.path.dirname(os.path.realpath(__file__))+"/../data/MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

learning_rate = 0.01
training_epochs = 1
batch_size = 100
display_step = 1

### modeling ###

activation = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

### training ###

for epoch in range(training_epochs) :

    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch) :

        batch_xs, batch_ys =mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch

    if epoch % display_step == 0 :
        print("Epoch : ", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Optimization Finished")

### predict number ###

r = random.randint(0, mnist.test.num_examples - 1)
print("Prediction: ", sess.run(tf.argmax(activation,1), {x: mnist.test.images[r:r+1]}))
print("Correct Answer: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))

image = np.zeros((1,784))
resized_image = cv2.resize(image, (28, 28))
sess.run(tf.argmax(activation,1), {x: resized_image})