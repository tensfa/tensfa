cm = tf.zeros(shape=[2,2], dtype=tf.int32)

for i in range(0, validation_data.shape[0], batch_size_validation):
    batched_val_data = np.array(validation_data[i:i+batch_size_validation, :, :], dtype='float')
    batched_val_labels = np.array(validation_labels[i:i+batch_size_validation, :], dtype='float')

    batched_val_data = batched_val_data.reshape((-1, n_chunks, chunk_size))

    _acc, _c, _p = sess.run([accuracy, correct, pred], feed_dict=({x:batched_val_data, y:batched_val_labels}))

    #batched_val_labels.shape ==> (2048, 2)
    #_p.shape                 ==> (2048, 2)
    #this piece of code throws the error!
    cm = tf.confusion_matrix(labels=batched_val_labels, predictions=_p)