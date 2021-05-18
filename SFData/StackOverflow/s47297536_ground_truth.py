import tensorflow as tf
import numpy as np

a  = np.zeros((1024, 1024, 3))
dtypes=[tf.float32]
shapes=[[1024, 3]]
q = tf.FIFOQueue(capacity=200,dtypes=dtypes,shapes=shapes)
enqueue_op = q.enqueue_many(a)
