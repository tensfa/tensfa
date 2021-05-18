import tensorflow as tf

idx1 = [1,2,3]
idx2 = [2,4,5]

idx1 = tf.expand_dims(idx1, 0)
idx2 = tf.expand_dims(idx2, 0)

intersection = tf.sets.intersection(tf.convert_to_tensor(idx1), tf.convert_to_tensor(idx2))
sess = tf.compat.v1.Session()
with sess.as_default():
  assert tf.compat.v1.get_default_session() is sess
  print(intersection.eval())