# Generate a Tensorflow Graph
tf.reset_default_graph()
batch_size = 25
embedding_size = 50
lstmUnits = 64
max_label = 2

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32, [None])

number_of_layers=3

#  Embeddings to represent words
# saved_embeddings = np.load('wordVectors.npy')
saved_embeddings = np.zeros((100, 10))
embeddings = tf.nn.embedding_lookup(saved_embeddings, x)

def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstmUnits,reuse=tf.get_variable_scope().reuse)

lstmCell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])

lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

outputs, final_state = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)

cost = tf.losses.mean_squared_error(y, predictions)