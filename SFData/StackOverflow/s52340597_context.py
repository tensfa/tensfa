
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets(os.path.dirname(os.path.realpath(__file__))+"/../data/MNIST_data/", one_hot=False)
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.train.images, mnist.train.labels

num_inputs = 28*28 # Size of images in pixels
num_hidden1 = 500
num_hidden2 = 500
num_outputs = 10 # Number of classes (labels)
learning_rate = 0.0011

inputs = tf.placeholder(tf.float32, shape=[None, num_inputs], name="x")
labels = tf.placeholder(tf.int32, shape=[None], name = "y")

def neuron_layer(x, num_neurons, name, activation=None):
    with tf.name_scope(name):
        num_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(num_inputs)
        init = tf.truncated_normal([num_inputs, num_neurons], stddev=stddev)
        W = tf.Variable(init, name = "weights")
        b = tf.Variable(tf.zeros([num_neurons]), name= "biases")
        z = tf.matmul(x, W) + b
        if activation == "sigmoid":
            return tf.sigmoid(z)
        elif activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(inputs, num_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, num_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, num_outputs, "output")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss)
    training_op = optimizer.apply_gradients(grads)

init = tf.global_variables_initializer()

num_epochs = 1
batch_size = 128

with tf.Session() as sess:
    init.run()
    print("Epoch\tTrain accuracy\tTest accuracy")
    for epoch in range(num_epochs):
        for idx_start in range(0, x_train.shape[0], batch_size):
            x_batch, y_batch = x_train[batch_size], y_train[batch_size]
            sess.run(training_op, feed_dict={inputs: x_batch, labels: y_batch})

        acc_train = sess.run(accuracy, feed_dict={inputs: x_train, labels: y_train})
        acc_test = sess.run(accuracy, feed_dict={inputs: x_test, labels: y_test})

        print("{}\t{}\t{}".format(epoch, acc_train, acc_test))