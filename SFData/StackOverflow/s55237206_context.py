import tensorflow as tf
import numpy as np

train_n = 10
test_n = 3
X_training = np.array(np.random.random((train_n, 100, 100, 3)), dtype=np.float32)
Y_training = np.eye(2)[np.random.randint(0, 2, train_n, dtype=np.int32)]

input = 100*100*3
batch_size = 25 #not used
X = tf.placeholder(tf.float32, [1, 100, 100, 3])
W = tf.Variable(tf.zeros([input, 2]))
b = tf.Variable(tf.zeros([2]))

init = tf.global_variables_initializer()
# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, input]), W) + b)
# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1 ))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)
sess = tf.Session()
sess.run(init)

for i in range(len(X_training)):
    # st = batch_size * i
    # end = st + batch_size - 1
    batch_X, batch_Y = X_training[i], Y_training[i]
    train_data={X: batch_X, Y_: batch_Y}

    sess.run(train_step, feed_dict=train_data)

    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
