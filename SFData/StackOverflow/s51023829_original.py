x_text = tf.placeholder(tf.float32, [None, *text.shape[1:]])

cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.ResidualWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=units))
    for units in [128, 256]
])

text_outputs, text_state = tf.nn.dynamic_rnn(
    cell=cells,
    inputs=x_text,
    dtype=tf.float32,
)