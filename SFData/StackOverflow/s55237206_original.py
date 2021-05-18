def simple_nn(X_training, Y_training, X_test, Y_test):
    input = 100*100*3
    batch_size = 25 #not used
    X = tf.placeholder(tf.float32, [1, 100, 100, 3])
    W = tf.Variable(tf.zeros([input, 2]))
    b = tf.Variable(tf.zeros([2]))

    init = tf.global_variables_initializer()
    # model
    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, input]), W) + b)
    # placeholder for correct labels
    Y_ = tf.placeholder(tf.float32, [None, 2])

    # loss function
    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

    # % of correct answers found in batch
    is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)
    sess = tf.Session()
    sess.run(init)

    for i in range(len(X_training)):
        # st = batch_size * i
        # end = st + batch_size - 1
        batch_X, batch_Y = X_training[i], Y_training[i]
        train_data={X: batch_X, Y_: batch_Y}

        sess.run(train_step, feed_dict=train_data)

        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

        # success on test data ?
    test_data={X: X_test, Y_: Y_test}
    a,c = sess.run([accuracy, cross_entropy], feed=test_data)