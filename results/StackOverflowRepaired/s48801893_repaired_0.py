import tensorflow as tf

DEPTH = 2

initia = tf.zeros_initializer
I = tf.placeholder(tf.float32, shape=[None,1], name='I') # input
W = tf.get_variable('W', shape=[1,DEPTH], initializer=initia, dtype=tf.float32) # weights
b = tf.get_variable('b', shape=[DEPTH], initializer=initia, dtype=tf.float32) # biases
O = tf.nn.relu(tf.matmul(I, W) + b, name='O')
O_0 = tf.gather_nd(O, [0,0])
W_1 = tf.gather_nd(W, [0,1])
distance = tf.square( O_0 - W_1 )
train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(distance)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(train_op, {I: [[1], [2], [3]]}))