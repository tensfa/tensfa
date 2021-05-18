# Create the cells
with tf.variable_scope('forward'):
    self.char_gru_cell_fw = tf.nn.rnn_cell.GRUCell(char_hidden_size)
with tf.variable_scope('backward'):
    self.char_gru_cell_bw = tf.nn.rnn_cell.GRUCell(char_hidden_size)

# Set initial state of the cells to be zero
self._char_initial_state_fw = \
    self.char_gru_cell_fw.zero_state(batch_size, tf.float32)
self._char_initial_state_bw = \
    self.char_gru_cell_bw.zero_state(batch_size, tf.float32)

#         Size before: batch-chrpad-chrvocabsize
#          Size after: batch-chrvocabsize
chargruinput = [tf.squeeze(input_, [1]) \
    for input_ in tf.split(1, char_num_steps, chargruinput)]

# Run the bidirectional rnn and get the corner results
_, output_state_fw, output_state_bw = \
   tf.nn.bidirectional_rnn(self.char_gru_cell_fw,
                    self.char_gru_cell_bw,
                    chargruinput,
                    sequence_length=char_num_steps,
                    initial_state_fw=self._char_initial_state_fw,
                    initial_state_bw=self._char_initial_state_bw)