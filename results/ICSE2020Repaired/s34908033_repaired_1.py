import tensorflow as tf
import numpy as np

data = np.array([0.1, 0.2])
data = np.expand_dims(data, -1)
x = tf.placeholder("float", shape=[2])
x = tf.expand_dims(x, 1)
T1 = tf.Variable(tf.ones([2,2]))
l1 = tf.matmul(T1, x)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    sess.run(l1, feed_dict={x: data})