cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batchSize =  128

epochs = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batches = batches(batchSize, train_features, train_labels)

    for epoch in range(epochs):
        for batch_features, batch_labels in total_batches:
            train_data = {features: batch_features, labels : batch_labels, keep_prob : 0.7}
            sess.run(optimizer, feed_dict = train_data)
        # Print status for every 100 epochs
        if epoch % 1000 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: val_features,
                    labels: val_labels,
                    keep_prob : 0.7})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
    Accuracy = sess.run(accuracy, feed_dict={features : test_features, labels :test_labels, keep_prob : 0.7})

    print('Trained Model Saved.')
print("Accuracy value is {}".format(Accuracy))