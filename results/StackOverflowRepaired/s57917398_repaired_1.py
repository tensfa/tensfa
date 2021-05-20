import tensorflow as tf

idx1 = [1,2,3]
idx2 = [2,4,5]

intersection = tf.sets.intersection(tf.expand_dims(tf.convert_to_tensor(idx1), 0), tf.expand_dims(tf.convert_to_tensor(idx2), 0))
sess = tf.compat.v1.Session()
with sess.as_default():
  assert tf.compat.v1.get_default_session() is sess
  print(intersection.eval())