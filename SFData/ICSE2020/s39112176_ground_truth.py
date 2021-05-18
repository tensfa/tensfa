import tensorflow as tf

init = tf.placeholder(tf.float32, (), name="init")
v = tf.Variable(tf.random_uniform((100, 300), -init, init), dtype=tf.float32)
initialize = tf.global_variables_initializer()

session = tf.Session()
session.run(initialize, feed_dict={init: 0.5})