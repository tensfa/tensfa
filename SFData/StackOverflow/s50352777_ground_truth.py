import tensorflow as tf
import numpy as np

def main():
    lstm_size = 128
    lstm_layers = 1
    batch_size = 50
    learning_rate = 0.001
    epochs = 10

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, None, 100], name='inputs')
        labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with graph.as_default():
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)

    with graph.as_default():
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs_, initial_state=initial_state)


    with graph.as_default():
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(labels_, predictions)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            state = sess.run(initial_state)
            x = np.zeros((batch_size, 10, 100), dtype='float32')
            y = np.zeros((batch_size, 1), dtype='int32')
            feed = {inputs_: x, labels_: y, keep_prob: 0.5, initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

if __name__ == '__main__':
    main()