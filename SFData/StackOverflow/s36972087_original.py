validation_images, validation_labels, train_images, train_labels = ld.read_data_set()
print "\n"
print len(train_images[0])
print len(train_labels)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 19])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([19,2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

start = 0
batch_1 = 50
end = 100

for i in range(1000):
  #batch = mnist.train.next_batch(50)
  x1 = train_images[start:end]
  y1 = train_labels[start:end]
  start = start + batch_1
  end = end + batch_1
  x1 = np.reshape(x1, (-1, 19))
  y1 = np.reshape(y1, (-1, 2))
  train_step.run(feed_dict={x: x1[0], y_: y1[0]})