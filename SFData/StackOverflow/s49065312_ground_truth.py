import tensorflow as tf
import numpy as np
from skimage import transform
tf.reset_default_graph()
from numpy import array
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets(os.path.dirname(os.path.realpath(__file__))+"/../data/MNIST_data")

x = tf.placeholder(tf.float32, [None, 484])

W = tf.get_variable("weights", shape=[484, 10],
                initializer=tf.random_normal_initializer())

b = tf.get_variable("bias", shape=[10],
                initializer=tf.random_normal_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

batch_size=50

for _ in range(1):
    batch_img, batch_label = mnist.train.next_batch(batch_size)
    batch_label = np.eye(10)[batch_label]

    imgs = batch_img.reshape((-1, 28, 28, 1))
    print(imgs.shape[0])

    resized_imgs = np.zeros((imgs.shape[0], 22, 22, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (22,22))
    image = array(resized_imgs, dtype=np.float32).reshape(imgs.shape[0], 484)
    print(image.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_step, feed_dict={x: image, y_: batch_label})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
