X = tf.placeholder(tf.float32, [batch_size, seq_len_1, 1], name='X')
labels = tf.placeholder(tf.float32, [None, alpha_size], name='labels')

rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
m_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * 3, state_is_tuple=True)
pre_prediction, state = tf.nn.dynamic_rnn(m_rnn_cell, X, dtype=tf.float32)