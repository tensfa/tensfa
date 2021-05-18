import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

train_n = 10
test_n = 5
learning_rate = 0.001
n_input = 2
n_steps = 1
n_hidden = 128
n_classes = 2

training_data = np.array(np.random.random((train_n, n_steps, n_input)), dtype=np.float32)
training_target = np.random.randint(0, n_classes, train_n, dtype=np.int32)
training_target = np.eye(n_classes)[training_target]

test_data = np.array(np.random.random((train_n, n_steps, n_input)),dtype=np.float32)
test_target = np.random.randint(0, n_classes, train_n, dtype=np.int32)
test_target = np.eye(n_classes)[test_target]

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    for i in range(len(training_data)):
        batch_x = training_data[i]
        batch_y = training_target[i]
        print(batch_x)
        print(batch_y)
        batch_x = tf.reshape(batch_x, [1, 2]).eval()
        print(batch_x)

        batch_x = np.expand_dims(batch_x, 0)
        batch_y = np.expand_dims(batch_y, 0)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_target}))