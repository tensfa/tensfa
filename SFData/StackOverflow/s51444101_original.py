from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

training_digits, training_labels = mnist.train.next_batch(1000)
test_digits, test_labels = mnist.test.next_batch(200)


height = 28
width = 28
channels = 1
n_inputs = height * width


conv1_feature_maps = 32
conv1_kernel_size = 3
conv1_stride = 1
conv1_pad = "SAME"


conv2_feature_maps = 64
conv2_kernel_size = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_feature_maps = conv2_feature_maps

n_fullyconn1 = 64
n_outputs = 10


tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])

y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_feature_maps,
                       kernel_size=conv1_kernel_size,
                       strides=conv1_stride, padding=conv1_pad,
                       activation = tf.nn.relu, name="conv1")


conv2 = tf.layers.conv2d(conv1, filters=conv2_feature_maps,
                       kernel_size=conv2_kernel_size,
                       strides=conv2_stride, padding=conv2_pad,
                       activation = tf.nn.relu, name="conv2")

pool3 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
pool3_flat = tf.reshape(pool3, shape=[-1,pool3_feature_maps * 7 *7])

fullyconn1 = tf.layers.dense(pool3_flat, n_fullyconn1, activation = tf.nn.relu, name="fc1")

logits = tf.layers.dense(fullyconn1, n_outputs, name="output")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                           labels = y)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 100

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)

        save_path = saver.save(sess, "./my_mnist_model")