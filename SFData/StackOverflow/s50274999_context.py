import tensorflow as tf

batched_val_labels = tf.zeros(shape=[2048, 2], dtype=tf.int32)
_p = tf.zeros(shape=[2048, 2], dtype=tf.int32)

cm = tf.confusion_matrix(labels=batched_val_labels, predictions=_p)
