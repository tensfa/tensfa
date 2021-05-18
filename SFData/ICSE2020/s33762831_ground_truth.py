import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq

input_sequence_length = 10
encoder_inputs = []
decoder_inputs = []
for i in range(350):
    encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_sequence_length],
                                              name="encoder{0}".format(i)))

for i in range(45):
    decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_sequence_length],
                                         name="decoder{0}".format(i)))

model = basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,rnn_cell.BasicLSTMCell(512))