import tensorflow as tf

tf.reset_default_graph()

num_inputs = 28*28 # Size of images in pixels
num_hidden1 = 500
num_hidden2 = 500
num_outputs = len(np.unique(y)) # Number of classes (labels)
learning_rate = 0.0011

inputs = tf.placeholder(tf.float32, shape=[None, num_inputs], name="x")
labels = tf.placeholder(tf.int32, shape=[None], name = "y")

print(np.expand_dims(inputs, axis=0))
print(np.expand_dims(labels, axis=0))

def neuron_layer(x, num_neurons, name, activation=None):
with tf.name_scope(name):
        num_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(num_inputs)
        init = tf.truncated_normal([num_inputs, num_neurons], stddev=stddev)
        W = tf.Variable(init, name = "weights")
        b = tf.Variable(tf.zeros([num_neurons]), name= "biases")
        z = tf.matmul(x, W) + b
    if activation == "sigmoid":
        return tf.sigmoid(z)
    elif activation == "relu":
        return tf.nn.relu(z)
    else:
        return z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(inputs, num_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, num_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, num_outputs, "output")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss)
    training_op = optimizer.apply_gradients(grads)

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

for grad, var in grads:
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradients", grad)

# summary
accuracy_summary = tf.summary.scalar('accuracy', accuracy)


# merge all summary
tf.summary.histogram('hidden1/activations', hidden1)
tf.summary.histogram('hidden2/activations', hidden2)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs/example03/dnn_final"
logdir = "{}/run-{}/".format(root_logdir, now)

train_writer = tf.summary.FileWriter("models/dnn0/train",
tf.get_default_graph())
test_writer = tf.summary.FileWriter("models/dnn0/test", tf.get_default_graph())

num_epochs = 50
batch_size = 128


with tf.Session() as sess:
    init.run()
    print("Epoch\tTrain accuracy\tTest accuracy")
    for epoch in range(num_epochs):
        for idx_start in range(0, x_train.shape[0], batch_size):
            idx_end = num_epochs
            x_batch, y_batch = x_train[batch_size], y_train[batch_size]
            sess.run(training_op, feed_dict={inputs: x_batch, labels: y_batch})

        summary_train, acc_train = sess.run([merged, accuracy],
                                       feed_dict={x: x_batch, y: y_batch})
        summary_test, acc_test = sess.run([accuracy_summary, accuracy],
                                     feed_dict={x: x_test, y: y_test})

        train_writer.add_summary(summary_train, epoch)
        test_writer.add_summary(summary_test, epoch)

        print("{}\t{}\t{}".format(epoch, acc_train, acc_test))

    save_path = saver.save(sess, "models/dnn0.ckpt")