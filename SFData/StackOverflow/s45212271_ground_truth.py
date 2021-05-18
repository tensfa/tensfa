import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6


n_classes = 56
batch_size = 1
hm_epochs = 1


#x = tf.placeholder('float', [None, 150528])
x = tf.placeholder('float', [None, 224,224,3])
y = tf.placeholder('float')


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    # size of window movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([56*56*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    #x = train_X

    #creating the first layer of CNN
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1']) # activation function 1
    conv1 = maxpool2d(conv1)

    #creating the second layer of CNN
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2']) # activation function 2
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 56*56*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output



def train_neural_network(x):
    i = 0

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_X)/batch_size)):
                _, c = sess.run([optimizer, cost], feed_dict={x: train_X[i:i+batch_size], y: train_y[i:i+batch_size]}) #HERE IS THE ERROR
                epoch_loss += c
                i += 100

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)


train_X = np.array(np.random.random((10, 224, 224, 3)), dtype=np.float32)
train_y = np.random.randint(0, n_classes, 10, dtype=np.int32)
train_y = np.eye(n_classes)[train_y]

train_neural_network(x)