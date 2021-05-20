import numpy as np
import tensorflow as tf

X_train = np.array(np.random.random((100, 10)), dtype=np.float32)
train_label = np.random.randint(0, 2, 100, np.int32)

X_test = np.array(np.random.random((100, 10)), dtype=np.float32)
test_label = np.random.randint(0, 2, 100, np.int32)

y_train = np.asarray(train_label)[:, None]
y_test = np.asarray(test_label)[:, None]

labels_train = (np.arange(2) == y_train[:,None]).astype(np.float32)
labels_train = labels_train.reshape(-1, 2)
labels_test = (np.arange(2) == y_test[:,None]).astype(np.float32)

inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
label = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

hid1_size = 128
w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)),  keep_prob=0.5)

hid2_size = 256
w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)

wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')
bo = tf.Variable(tf.random_normal([2, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

pred = tf.nn.softmax(yo)
pred_label = tf.argmax(pred, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
sess.run(init)

for learning_rate in [0.05, 0.01]:
    for epoch in range(1):
        avg_cost = 0.0
        for i in range(X_train.shape[0]):
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate,
                                                          inputs: X_train[i, None],
                                                          label: labels_train[i, None].reshape(-1, 2)})
            avg_cost += c
        avg_cost /= X_train.shape[0]
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: X_test, label: labels_test})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))

sess.close()