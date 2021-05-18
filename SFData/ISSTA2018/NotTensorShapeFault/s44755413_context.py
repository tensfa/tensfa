import tensorflow as tf

zero_tsr = tf.zeros([1, 2])
op = tf.assign(zero_tsr, [4, 5])
sess = tf.Session()
_ = sess.run(op)
print(sess.run(zero_tsr))