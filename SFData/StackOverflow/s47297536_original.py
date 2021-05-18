a  = np.zeros((1024, 1024, 3))
dtypes=[tf.float32]
print len(dtypes)
shapes=[1024, 1024, 3]
print len(shapes)
q = tf.FIFOQueue(capacity=200,dtypes=dtypes,shapes=shapes)