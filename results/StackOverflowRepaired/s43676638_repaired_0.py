import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
batch_size = 64

#coding:utf-8
def rnn_model(x, weights, biases):
    """RNN (LSTM or GRU) model for image"""
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, 0)
    # Define a lstm cell with tensorflow
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

def main():
    """Train an image classifier"""
    """Step 0: load image data and training parameters"""
    mnist = input_data.read_data_sets(os.path.dirname(os.path.realpath(__file__))+"/../data/MNIST_data/", one_hot=False)# change one_hot to false by myself ---------

    """Step 1: build a rnn model for image"""
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    #y = tf.placeholder("float", [n_classes,])

    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
    biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

    pred = rnn_model(x, weights, biases)
    # Define loss and optimizer
    #you will get the dreaded 'No gradients provided for any variable' if you switch the args between y and pred
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    """Step 2: train the image classification model"""
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())# deprecated
        sess.run(tf.global_variables_initializer())
        step = 1

        """Step 2.1: train the image classifier batch by batch"""
        while step * batch_size < 10000:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_y = np.eye(10)[batch_y]
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})#**bug is here**

            """Step 2.2: save the model"""
            if step % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(step * batch_size, loss, acc))
            step += 1
        print("The training is done")

        """Step 3: test the model"""
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

if __name__ == '__main__':
    main()