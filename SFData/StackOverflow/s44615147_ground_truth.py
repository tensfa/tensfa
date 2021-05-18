import tensorflow as tf

batch_size = 64
seq_len_1 = 10
alpha_size  = 3

X = tf.placeholder(tf.float32, [batch_size, seq_len_1, 1], name='X')
labels = tf.placeholder(tf.float32, [None, alpha_size], name='labels')

stacked_rnn = []
for iiLyr in range(3):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True))
m_rnn_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)
pre_prediction, state = tf.nn.dynamic_rnn(m_rnn_cell, X, dtype=tf.float32)