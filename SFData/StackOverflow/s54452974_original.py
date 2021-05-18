import tensorflow as tf
import numpy as np

x = np.array([1.0, 1.0, 1.0])
z = tf.ones((1, 3))

out = tf.ones((1, 3))
print('out:', out)
i = tf.constant(0)

def cond(i, _):
    return i < 10


def body(i, out):
    i = i + 1
    out = tf.concat([out, out], axis=0)
    return [i, out]

_, out = tf.while_loop(cond, body, [i, out], shape_invariants=[i.get_shape(), tf.TensorShape([None])])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
res = sess.run([_, out])
print(res)