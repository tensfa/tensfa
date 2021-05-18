dataet=[[[3, 5], [7, 2], [7, 6]],
        [[2, 5], [1, 3], [4, 3]],
        [[8, 1], [1, 8], [9, 3]],
        [[1, 5], [6, 7], [4, 9]]]



import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


input_x=tf.placeholder(dtype=tf.int32,shape=[4,3,2])
input_x=tf.cast(input_x,tf.float32)





data=tf.unstack(input_x,3,axis=1)


with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(num_units=250)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=data,dtype=tf.float32)

output,(fs,fc)=model







with tf.Session() as sess:
    unstack_output,output_n=sess.run([output,data],feed_dict={input_x:dataet})
    print(unstack_output,output_n)