import tensorflow as tf
import numpy as np

class Deep_Autoencoder:
    def __init__(self, input_dim, n_nodes_hl = (32, 16, 1), epochs = 1, batch_size = 128, learning_rate = 0.02, n_examples = 10):
        # Hyperparameters
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_examples = n_examples

        # Input and target placeholders
        X = tf.placeholder('float', [None, self.input_dim])
        Y = tf.placeholder('float', [None, self.input_dim])
        ...

        self.X = X
        print("self.X : ", self.X)
        self.Y = Y
        print("self.Y : ", self.Y)
        ...

    def train_neural_network(self, data, targets):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                epoch_loss = 0
                i = 0
                # Let's train it in batch-mode
                while i < len(data):
                    start = i
                    end = i + self.batch_size

                    batch_x = np.array(data[start:end])
                    batch_y = np.array(targets[start:end])

                    batch_x = np.expand_dims(batch_x, 1)
                    batch_y = np.expand_dims(batch_y, 1)

                    # batch_x = np.expand_dims(batch_x)
                    sess.run([self.X, self.Y], feed_dict={self.X: batch_x, self.Y: batch_y})
                    i += self.batch_size

train_n = 10
n_input = 1

training_data = np.array(np.random.random(train_n), dtype=np.float32)
training_target = np.random.randint(0, n_input, train_n, dtype=np.int32)

da = Deep_Autoencoder(n_input)
da.train_neural_network(training_data, training_target)