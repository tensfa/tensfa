import tensorflow as tf
import numpy as np
i = tf.constant(0)
a = tf.constant(1, shape = [1], dtype = np.float32)

def che(i, a):
    return tf.less(i, 10)

def ce(i, a):
    a = tf.concat([a, tf.constant(2, shape = [1], dtype = np.float32)], axis = -1)
    i = tf.add(i, 1)
    return i, a
c, p = tf.while_loop(che, ce, loop_vars = [i, a], shape_invariants = [i.get_shape(), tf.TensorShape([None,])])