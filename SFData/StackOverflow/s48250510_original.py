# placeholders for the data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weights and biases
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax model
activation = tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, w) + b, labels=y)
# backpropagation
train = tf.train.GradientDescentOptimizer(0.5).minimize(activation)

# creating tensorflow session
s = tf.InteractiveSession()

# i have already initialised the variables

# gradient descent
for i in range(100):
    x_bat, y_bat= create_batch(x_train, y_train, size=100)
    train_step = s.run(train, feed_dict={x: x_bat, y: y_bat})