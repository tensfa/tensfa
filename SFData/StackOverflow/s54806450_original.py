 X_train,X_test,y_train,y_test = train_test_split(final_training_set, labels, test_size=0.2, shuffle=False, random_state=42)


epochs = 10
time_steps = 555
n_classes = 2
n_units = 128
n_features = 9
batch_size = 8

x= tf.placeholder('float32',[batch_size,time_steps,n_features])
y = tf.placeholder('float32',[None,n_classes])

###########################################
out_weights=tf.Variable(tf.random_normal([n_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))
###########################################

lstm_layer=tf.nn.rnn_cell.LSTMCell(n_units,state_is_tuple=True)
initial_state = lstm_layer.zero_state(batch_size, dtype=tf.float32)
outputs,states = tf.nn.dynamic_rnn(lstm_layer, x,
                                   initial_state=initial_state,
                                   dtype=tf.float32)


###########################################
output=tf.matmul(outputs[-1],out_weights)+out_bias
print(np.shape(output))

logit = output
logit = (logit, [-1])

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(cost)
with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for epoch in range(epochs):
            epoch_loss = 0

            i = 0
            for i in range(int(len(X_train) / batch_size)):

                start = i
                end = i + batch_size

                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                epoch_loss += c

                i += batch_size

            print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

        pred = tf.round(tf.nn.sigmoid(logit)).eval({x: np.array(X_test), y: np.array(y_test)})

        f1 = f1_score(np.array(y_test), pred, average='macro')

        accuracy=accuracy_score(np.array(y_test), pred)


        print("F1 Score:", f1)
        print("Accuracy Score:",accuracy)