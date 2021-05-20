import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Imports
import tensorflow as tf
import numpy as np

X_train = np.array(np.random.random((1000, 25, 4)), dtype=np.float32)
y_train = np.eye(2)[np.random.randint(0,2,1000,dtype=np.int32)]

#split data into train / validation and test
X_input = X_train[0:900]
y_input = y_train[0:900]

#print (X_input.shape)
#print (y_input.shape)

X_train_data = X_input[0:630]
X_test_data = X_input[630:900]

y_train_data = y_input[0:630]
y_test_data = y_input[630:900]

# Variables
hidden_layer_1_nodes = 300
hidden_layer_2_nodes = 100
output_layer_nodes = 100
epochs = 10
classes = 2
epoch_errors = []
stddev = 0.035
learning_rate = 0.08
batch_size = 100

#print (X_train_data[0])

# TF Placeholders
X = tf.placeholder('float32', [None, 100], name='X')
y = tf.placeholder('float32', [None, classes], name='y')

# Weights Matrices
W1 = tf.Variable(tf.truncated_normal([100, hidden_layer_1_nodes], stddev=stddev), name='W1')
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1_nodes, hidden_layer_2_nodes], stddev=stddev), name='W2')
W3 = tf.Variable(tf.truncated_normal([hidden_layer_2_nodes, output_layer_nodes], stddev=stddev), name='W3')
W4 = tf.Variable(tf.truncated_normal([output_layer_nodes, classes], stddev=stddev), name='W4')

# Biases Vectors
b1 = tf.Variable(tf.truncated_normal([hidden_layer_1_nodes], stddev=stddev), name='b1')
b2 = tf.Variable(tf.truncated_normal([hidden_layer_2_nodes], stddev=stddev), name='b2')
b3 = tf.Variable(tf.truncated_normal([output_layer_nodes], stddev=stddev), name='b3')
b4 = tf.Variable(tf.truncated_normal([classes], stddev=stddev), name='b4')

# Define the Neural Network
def nn_model(X):
    input_layer     =    {'weights': W1, 'biases': b1}
    hidden_layer_1  =    {'weights': W2, 'biases': b2}
    hidden_layer_2  =    {'weights': W3, 'biases': b3}
    output_layer    =    {'weights': W4, 'biases': b4}

    input_layer_sum = tf.add(tf.matmul(X, input_layer['weights']), input_layer['biases'])
    input_layer_sum = tf.nn.relu(input_layer_sum)

    hidden_layer_1_sum = tf.add(tf.matmul(input_layer_sum, hidden_layer_1['weights']), hidden_layer_1['biases'])
    hidden_layer_1_sum = tf.nn.relu(hidden_layer_1_sum)

    hidden_layer_2_sum = tf.add(tf.matmul(hidden_layer_1_sum, hidden_layer_2['weights']), hidden_layer_2['biases'])
    hidden_layer_2_sum = tf.nn.relu(hidden_layer_2_sum)

    output_layer_sum = tf.add(tf.matmul(hidden_layer_2_sum, output_layer['weights']), output_layer['biases'])
    return output_layer_sum

# Train the Neural Network
def nn_train(X):
    pred = nn_model(X)
    pred = tf.identity(pred)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        #saver = tf.train.Saver()
        sess.run(init_op)

        for epoch in range(epochs):
            epoch_loss = 0.0

            i = 0
            while i < len(X_train_data):
                start = i
                end = i+batch_size

                batch_x = np.array(X_train_data[start:end])
                batch_x = batch_x.reshape(-1, 100)
                batch_y = np.array(y_train_data[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, y: batch_y})
                epoch_loss += c
                i+= batch_size

            epoch_errors.append(epoch_loss)
            print('Epoch ', epoch + 1, ' of ', epochs, ' with loss: ', epoch_loss)

        correct_result = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_result, 'float'))
        print('Acc: ', accuracy.eval({X:X_test_data, y:y_test_data}))


def main():
    nn_train(X)


if __name__ == "__main__":
    main()