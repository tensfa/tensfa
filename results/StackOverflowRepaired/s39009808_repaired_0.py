import tensorflow as tf

batch_size = 32
chrvocabsize = 100
char_hidden_size = 64
char_num_steps = 10

# Create the cells
with tf.variable_scope('forward'):
    char_gru_cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden_size)
with tf.variable_scope('backward'):
    char_gru_cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden_size)

# Set initial state of the cells to be zero
_char_initial_state_fw = \
    char_gru_cell_fw.zero_state(batch_size, tf.float32)
_char_initial_state_bw = \
    char_gru_cell_bw.zero_state(batch_size, tf.float32)

chargruinput = tf.Variable(tf.random_normal(shape=[batch_size, char_num_steps, chrvocabsize]),dtype=tf.float32)

# Run the bidirectional rnn and get the corner results
output_state_fw, output_state_bw = \
   tf.nn.bidirectional_dynamic_rnn(char_gru_cell_fw,
                    char_gru_cell_bw,
                    inputs=chargruinput,
                    sequence_length=[char_num_steps]*batch_size,
                    initial_state_fw=_char_initial_state_fw,
                    initial_state_bw=_char_initial_state_bw)