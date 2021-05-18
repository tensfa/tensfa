import tensorflow as tf
images=tf.reshape(tf.range(100*100*3*5), [100, 100, 3, 5])
batch_crop = tf.random_crop(images, size=(20, 20, 3, 5))
with tf.Session() as sess:
     batch = sess.run([batch_crop])