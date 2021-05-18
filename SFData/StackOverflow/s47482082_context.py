import tensorflow as tf
import numpy as np

print("def layers")
x = tf.placeholder(tf.float32, [ None, 32*32 ])
y_ = tf.placeholder(tf.float32, [None, 5 ])

W = tf.Variable(tf.zeros([ 32*32, 5 ]))
b = tf.Variable(tf.zeros([ 5 ]))
y = tf.matmul(x, W) + b

print("def leraning model")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction= tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

imgs = np.array(np.random.random((100, 32*32)), dtype=np.float32)
labels = np.random.randint(0, 5, 100, np.int32)

with tf.Session() as sess:
    print("init layer value")
    sess.run(tf.global_variables_initializer())
    for i in range(0, 10):
        print("train num %d" % (i+1))
        sess.run([train_step, accuracy], feed_dict={x:imgs, y_: labels})