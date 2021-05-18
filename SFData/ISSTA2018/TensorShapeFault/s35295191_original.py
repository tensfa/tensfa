def multilayer_perceptron(_X, _weights, _biases):
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
        embedding_layer = tf.nn.embedding_lookup(W, _X)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(embedding_layer, _weights['h1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return tf.matmul(layer_2, weights['out']) + biases['out']

x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.initialize_all_variables()