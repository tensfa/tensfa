import tensorflow as tf

X = tf.placeholder(shape=(1, 5, 7), name='inputs', dtype=tf.float32)
y = tf.placeholder(shape=(1), name='outputs', dtype=tf.int32)

flattened = tf.layers.flatten(X) # shape (1,35)
hidden1 = tf.layers.dense(flattened, 150) # shape (1,150)
hidden2 = tf.layers.dense(hidden1, 50) # shape (1,50)
logits = tf.layers.dense(hidden2, 1) # shape (1,1)

with tf.name_scope("loss"):
      # expects logits of shape (1,1) against labels of shape (1)
      xentropy= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                   logits=logits)
      loss = tf.reduce_mean(xentropy, name="loss")