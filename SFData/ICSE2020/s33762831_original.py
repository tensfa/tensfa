import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

encoder_inputs = []
decoder_inputs = []
for i in xrange(350):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name="encoder{0}".format(i)))

for i in xrange(45):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                         name="decoder{0}".format(i)))

model = seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                  decoder_inputs,rnn_cell.BasicLSTMCell(512))