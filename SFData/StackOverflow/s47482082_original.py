import os
import random

import tensorflow as tf

import load_datasets
import datasets
import make_datasets

print("def input and output")
images = tf.placeholder(tf.float32, shape=[None, 32*32])
labels = tf.placeholder(tf.int32, shape=[None, 5])


print("def layers")
x = tf.placeholder(tf.float32, [ None, 32*32 ])
y_ = tf.placeholder(tf.float32, [None, 5 ])

# W1 = tf.Variable(tf.zeros([ 32*32, 500 ]))
# b1 = tf.Variable(tf.zeros([ 500 ]))

# W2 = tf.Variable(tf.zeros([ 500, 5 ]))
# b2 = tf.Variable(tf.zeros([ 5 ]))

print("def function")
# h1 = tf.matmul(x, W1) + b1
# y = tf.matmul(h1, W2) + b2

W = tf.Variable(tf.zeros([ 32*32, 5 ]))
b = tf.Variable(tf.zeros([ 5 ]))
y = tf.matmul(x, W) + b

print("def leraning model")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction= tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("load train dataset")
trainfilepath = "../03tfrecords/train.tfrecord"
images, labels = load_datasets.read_tfrecord(trainfilepath)
input_queue = tf.train.slice_input_producer( [images, labels ], num_epochs=10, shuffle=False )
image_batch, label_batch = tf.train.batch( [images, labels], batch_size=10)

print("load test dataset")
testfilepath = "../03tfrecords/test.tfrecord"
test_image, test_label = load_datasets.read_tfrecord(testfilepath)
img_test_batch, label_test_batch = tf.train.batch([test_image,test_label],batch_size=16)

with tf.Session() as sess:
    print("init layer value")
    sess.run(tf.global_variables_initializer())
    print("start training")
    tf.train.start_queue_runners(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            for i in range(0, 10):
                print("train num %d" % (i+1))
                imgs, labels = sess.run([image_batch, label_batch])
                sess.run(train_step, feed_dict={x:imgs, y_: labels})

                imgs_test, labels_text = sess.run([img_test_batch, label_test_batch])
                print(sess.run(accuracy, feed_dict={x:imgs_test, y_:labels_text}))


    finally:
        coord.request_stop()
        coord.join(threads)