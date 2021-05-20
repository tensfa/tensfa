import tensorflow as tf
import numpy as np

train_images = np.array(np.random.random((10, 19)), dtype=np.float32)
train_labels = np.random.randint(0, 2, 10, dtype=np.int32)
train_labels = np.eye(2)[train_labels]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 19])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([19,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

start = 0
batch_1 = 50
end = 100

for i in range(1):
  # batch = mnist.train.next_batch(50)
  x1 = train_images[start:end]
  y1 = train_labels[start:end]
  start = start + batch_1
  end = end + batch_1
  x1 = np.reshape(x1, (-1, 19))
  y1 = np.reshape(y1, (-1, 2))
  train_step.run(feed_dict={x: np.expand_dims(x1[0], 0), y_: np.expand_dims(y1[0], 0)})