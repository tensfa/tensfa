import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x1 = tf.placeholder(tf.float32, shape=[None,20])
Wo1 = weight_variable([20, 1])
bo1 = bias_variable([1])
tf.matmul(x1, Wo1) + bo1
