# the input shape is (batch_size, input_size)
input_size = tf.shape(input_data)[1]

# labels in one-hot format have shape (batch_size, num_classes)
num_classes = tf.shape(labels)[1]

stddev = 1.0 / tf.cast(input_size, tf.float32)

w_shape = tf.pack([input_size, num_classes], 'w-shape')
normal_dist = tf.truncated_normal(w_shape, stddev=stddev, name='normaldist')
self.w = tf.Variable(normal_dist, name='weights')