import tensorflow as tf

batched_val_labels = tf.zeros(shape=[2048, 2], dtype=tf.int32)
_p = tf.zeros(shape=[2048, 2], dtype=tf.int32)

batched_val_labels = tf.argmax(batched_val_labels, 1)
_p = tf.argmax(_p, 1)

# labels	1-D Tensor of real labels for the classification task.
# predictions	1-D Tensor of predictions for a given classification.
cm = tf.confusion_matrix(labels=batched_val_labels, predictions=_p)
