import tensorflow as tf
import numpy as np
import pandas as pd
import math

df1=pd.read_csv(r'C:\Ocean of knowledge\Acads\7th sem\UGP\datasets\xTrain.csv')
df1 = df1.dropna()
xTrain = df1.values


df2 = pd.read_csv(r'C:\Ocean of knowledge\Acads\7th sem\UGP\datasets\yTrain.csv')
df2 = df2.dropna()
yTrain = df2.values

sess=tf.Session()
saver = tf.train.import_meta_graph(r'C:\Ocean of knowledge\Acads\7th sem\UGP\NeuralNet\my_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("input:0")
feed_dict ={w1:xTrain1}
op_to_restore = graph.get_tensor_by_name("hidden:0")
h1 = sess.run(op_to_restore,feed_dict)
print(h1)

n_input1 = 20
n_hidden1 = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x1 = tf.placeholder(tf.float32, shape=[])
Wo1 = weight_variable([20,1])
bo1 = bias_variable([1])
y1 = tf.nn.tanh(tf.matmul((x1,Wo1)+ bo1),name="op_to_restore2_")

y1_ = tf.placeholder("float", [None,n_hidden1], name="check_")
meansq1 = tf.reduce_mean(tf.square(y1- y1_), name="hello_")
train_step1 = tf.train.AdamOptimizer(0.005).minimize(meansq1)

#init = tf.initialize_all_variables()

init = tf.global_variables_initializer()
sess.run(init)

n_rounds1 = 100
batch_size1 = 5
n_samp1 = 350

for i in range(n_rounds1+1):
    sample1 = np.random.randint(n_samp1, size=batch_size1)
    batch_xs1 = h1[sample1][:]
    batch_ys1 = yTrain[sample1][:]
    sess.run(x1, feed_dict={x1: batch_xs1, y1_:batch_ys1})