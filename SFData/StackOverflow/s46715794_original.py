class Deep_Autoencoder:
    def __init__(self, input_dim, n_nodes_hl = (32, 16, 1), epochs = 400, batch_size = 128, learning_rate = 0.02, n_examples = 10):

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
                print("type batch_x :", type(batch_x))
                print("len batch_x :", len(batch_x))
                batch_y = np.array(targets[start:end])
                print("type batch_y :", type(batch_y))
                print("len batch_y :", len(batch_y))

                hidden, _, c = sess.run([self.encoded, self.optimizer, self.cost], feed_dict={self.X: batch_x, self.Y: batch_y})
                epoch_loss +=c
                i += self.batch_size

        self.saver.save(sess, 'selfautoencoder.ckpt')
        print('Accuracy', self.accuracy.eval({self.X: data, self.Y: targets}))