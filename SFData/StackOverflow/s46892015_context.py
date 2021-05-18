import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np

DNA_SIZE = 1
POP_SIZE = 10
LR = 0.1
N_GENERATION = 50

def F(x):
    return x**2

def get_fitness(value):
    return -value

mean = tf.Variable(tf.constant(13.), dtype=tf.float32)
sigma = tf.Variable(tf.constant(5.), dtype=tf.float32)
N_dist = Normal(loc=mean, scale=sigma)
make_kids = N_dist.sample([POP_SIZE])

tfkids = tf.placeholder(tf.float32, [POP_SIZE, DNA_SIZE])
tfkids_fit = tf.placeholder(tf.float32, [POP_SIZE])
loss = -tf.reduce_mean(N_dist.log_prob(tfkids) * tfkids_fit)
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

kids = sess.run(make_kids)
kids_fit = get_fitness(F(kids))
sess.run(train_op, feed_dict={tfkids: kids, tfkids_fit: kids_fit})
