import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

dataset = np.array(np.random.random((4,3,2)), dtype=np.float32)

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(num_units=250)
    model=tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs=dataset,dtype=tf.float32)

output,(fs,fc)=model
