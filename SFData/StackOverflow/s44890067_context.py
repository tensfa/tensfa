import tensorflow as tf
import numpy as np

n_inputs = 5000
n_classes = 1161
features = tf.placeholder(tf.float32, [None, n_inputs])
labels = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

h_layer = 256

weights = {
'hidden_weights' : tf.Variable(tf.random_normal([n_inputs, h_layer])),
'out_weights' : tf.Variable(tf.random_normal([h_layer, n_classes]))
}

bias = {
'hidden_bias' : tf.Variable(tf.random_normal([h_layer])),
'out_bias' : tf.Variable(tf.random_normal([n_classes]))
}

hidden_output1 = tf.add(tf.matmul(features, weights['hidden_weights']),bias['hidden_bias'])
hidden_relu1 = tf.nn.relu(hidden_output1)
hidden_out = tf.nn.dropout(hidden_relu1, keep_prob)

hidden_output2 = tf.add(tf.matmul(hidden_out, weights['out_weights']),bias['out_bias'])
logits = tf.nn.relu(hidden_output2)
logits = tf.nn.dropout(logits, keep_prob)
learn_rate = 0.001


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batchSize =  10

epochs = 1
init = tf.global_variables_initializer()

def batches(bs, f, l):
    fls = []
    i = 0
    while i + bs <= len(f):
        fls.append((f[i:i+bs], l[i:i+bs]))
        i += bs
    return fls

train_n = 10
val_n = 10
test_n = 10

train_features = np.array(np.random.random((train_n, n_inputs)),dtype=np.float32)
train_labels = np.random.randint(0, n_classes, train_n, dtype=np.int32)
val_features = np.array(np.random.random((val_n, n_inputs)),dtype=np.float32)
val_labels = np.random.randint(0, n_classes, val_n, dtype=np.int32)
test_features = np.array(np.random.random((test_n, n_inputs)),dtype=np.float32)
test_labels = np.random.randint(0, n_classes, test_n, dtype=np.int32)

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