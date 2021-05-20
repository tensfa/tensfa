import tensorflow as tf
import numpy as np


image_to_use = np.array(np.random.random((128, 128, 3)), dtype=np.float32)
image_to_use = np.expand_dims(image_to_use, 0)
print(image_to_use.shape)

with tf.Session() as sess:

    keep_prob_tf = tf.placeholder(tf.float32, name="keep-prob-in")
    image_in_tf = tf.placeholder(tf.float32, [None, image_to_use.shape[0], image_to_use.shape[1], image_to_use.shape[2]], name="image-in")

    units = sess.run(image_in_tf, feed_dict={image_in_tf:image_to_use, keep_prob_tf:1.0})